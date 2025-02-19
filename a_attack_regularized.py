from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def cos_sim(x, y):
    # Normalize the tensors
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)
    
    # Compute the dot product
    dot_product = torch.sum(x * y)
    
    # Compute the cosine similarity
    cosine_similarity = dot_product / (norm_x * norm_y)
    
    return cosine_similarity

def count_common(x, y):
    num_common = 0
    for elem in x:
        if  elem in y:
            num_common += 1
    return num_common

def activation_weighted_cosine(sae_out, z_target):
    weights = torch.abs(z_target)  # Assign higher weight to important activations
    cosine_sim = torch.nn.functional.cosine_similarity(sae_out, z_target, dim=-1)
    return -(weights * cosine_sim).sum() / weights.sum()

def l1_activation_loss(sae_out, z_target):
    return torch.nn.functional.l1_loss(sae_out, z_target)

def contrastive_overlap_loss(sae_out, z_target, top_k=192, threshold=0.1):
    top_indices = torch.topk(z_target, top_k).indices  # Get important neurons
    mask = torch.ones_like(sae_out, dtype=torch.bool)
    mask[top_indices] = False  # Ignore top activations, penalize others
    penalty = torch.nn.functional.relu(sae_out - threshold) * mask
    return penalty.sum()

data_file = "./sae_samples_50.csv"
df = pd.read_csv(data_file)
sample_idx = 16
layer_num = 20
sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
src_text = df.iloc[sample_idx]['x1']
target_text = df.iloc[sample_idx]['x2']
print(f"x1: {src_text}")
print(f"x2: {target_text}")

x_src = tokenizer(src_text, return_tensors="pt").to(DEVICE)
x_target = tokenizer(target_text, return_tensors="pt").to(DEVICE)

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)

h_src = model(**x_src, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
h_target = model(**x_target, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]

z_src = sae.pre_acts(h_src)
z_target = sae.pre_acts(h_target)
top_idx_src = sae.encode(h_src).top_indices
top_idx_target = sae.encode(h_target).top_indices
initial_overlap = count_common(top_idx_src, top_idx_target) / len(top_idx_src)
print(f"Initial overlap ratio = {initial_overlap}")

num_iters = 7
k = 500
lambda_1, lambda_2, lambda_3 = 1.0, 0.0, 0.0  # Tune these weights

x_src = torch.tensor(tokenizer.encode(src_text)).unsqueeze(0).to(DEVICE)
x_src_old = x_src.clone()

model.to(DEVICE)
best_loss = 100.0

similarities = []
overlaps = []
overlap_increases = []
overlap_increase_ratios = []

print(f"Original Input: {tokenizer.decode(x_src[0], skip_special_tokens=True)}")

for t in range(1, x_src.shape[-1]):
    x_src = x_src_old
    best_loss = 100.0

    for i in range(num_iters):
        with torch.no_grad():
            out = model(x_src, output_hidden_states=True)
        embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
        lm_out = model(inputs_embeds=embeddings, output_hidden_states=True)
        sae_out = sae.pre_acts(lm_out.hidden_states[layer_num + 1][0][-1])
        # print(torch.max(sae_out), torch.min(sae_out))
        # print(sae_out[:100])
        # print(sae.encode(lm_out.hidden_states[layer_num + 1][0][-1]).top_acts)
        # print(sae.cfg.signed)

        # Combine the losses with adjustable weights
        loss = (
            lambda_1 * activation_weighted_cosine(sae_out, z_target) +
            lambda_2 * l1_activation_loss(sae_out, z_target) +
            lambda_3 * contrastive_overlap_loss(sae_out, z_target)
        )

        gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
        dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)

        cls_token_idx = tokenizer.encode('[CLS]')[1]
        sep_token_idx = tokenizer.encode('[SEP]')[1]
        dot_prod[:, cls_token_idx] = -float('inf')
        dot_prod[:, sep_token_idx] = -float('inf')

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, k).indices)[t]   
        tokens_batch = []

        for k_idx in range(k):
            batch_item = x_src.clone().detach()
            batch_item[0, t] = top_k_adv[k_idx]
            tokens_batch.append(batch_item)

        tokens_batch = torch.cat(tokens_batch, dim=0)

        with torch.no_grad():
            new_embeds = model(tokens_batch, output_hidden_states=True).hidden_states[0]
            out = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num + 1]
            out = sae.pre_acts(out[:, -1, :])
        loss_batch = torch.tensor([activation_weighted_cosine(out[j], z_target) for j in range(out.shape[0])])
        
        best_idx = torch.argmin(loss_batch)
        if loss_batch[best_idx] < best_loss:
            best_loss = loss_batch[best_idx]
            best_similarity = -best_loss.item()
            x_src = tokens_batch[best_idx].unsqueeze(0)
        
        with torch.no_grad():
            out = model(x_src, output_hidden_states=True)
        top_idx_src = sae.encode(out.hidden_states[layer_num + 1][0][-1]).top_indices
        num_overlap = count_common(top_idx_src, top_idx_target)
        overlap = num_overlap / len(top_idx_target)
    
        print(f"Iteration {i+1} similarity = {best_similarity}")    
        print(f"Iteration {i+1} num_overlap = {num_overlap}, overlap_ratio={overlap}")   
        
        print(f"Iteration {i+1} input: {tokenizer.decode(x_src[0], skip_special_tokens=True)}")
        print("--------------------")
    print(f"Token {t} best similarity: {best_similarity}")
    print(f"Token {t} best overlap: {overlap}")
    overlap_increase = overlap - initial_overlap
    overlap_increase_ratio = overlap_increase / initial_overlap
    print(f"Token {t} overlap increase: {overlap_increase}, {overlap_increase_ratio}")
    similarities.append(best_similarity)
    overlaps.append(overlap)
    overlap_increases.append(overlap_increase)
    overlap_increase_ratios.append(overlap_increase_ratio)

print(f"All similarities: {similarities}")
print(f"All overlaps: {overlaps}")
print(f"All overlap increases: {overlap_increases}")
print(f"All overlap increase ratios: {overlap_increase_ratios}")