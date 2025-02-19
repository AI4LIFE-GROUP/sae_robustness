import sys
sys.path.append('../sae')
from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

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

attack_start_time = time.time()
data_file = "./sae_samples_50.csv"
df = pd.read_csv(data_file)
sample_idx = 40
layer_num = 20
neuron_idx = 100
maximize = True
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

z_src = sae.pre_acts(h_src)
top_idx_src = sae.encode(h_src).top_indices
top_acts_src = sae.encode(h_src).top_acts

num_iters = 7
k = 200

x_src = torch.tensor(tokenizer.encode(src_text)).unsqueeze(0).to(DEVICE)
x_src_old = x_src.clone()

model.to(DEVICE)
best_loss = 100.0

losses = []
overlaps = []
print(f"Original Input: {tokenizer.decode(x_src[0], skip_special_tokens=True)}")

for t in range(1, x_src.shape[-1]):
    x_src = x_src_old
    best_loss = float('inf')
    best_overlap = 1.0

    for i in range(num_iters):
        with torch.no_grad():
            out = model(x_src, output_hidden_states=True)
        embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
        lm_out = model(inputs_embeds=embeddings, output_hidden_states=True)
        sae_out = sae.pre_acts(lm_out.hidden_states[layer_num + 1][0][-1])
        # drift = torch.norm(sae_out - z_src, p=1)
        cosine_loss = cos_sim(sae_out, z_src)
        # print(drift, cosine_loss)
        loss = -cosine_loss 
        # loss = torch.norm(sae_out - z_src, p=2)
        gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0] # (1, seq_len, lm_dim)
        # gradients = gradients / (torch.norm(gradients, dim=-1, keepdim=True) + 1e-8)
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
            h_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num + 1] # (k, seq_len, lm_dim)
            z_batch = sae.pre_acts(h_batch[:, -1, :]) # (k, sae_dim)
        top_idx_batch = sae.encode(h_batch[:, -1, :]).top_indices # (k, num_topk=192)
        overlap_batch = torch.tensor([count_common(top_idx_batch[j], top_idx_src) / len(top_idx_src) for j in range(top_idx_batch.shape[0])])
        # loss_batch = torch.tensor([cos_sim(out[j], z_src) - 0.005 * torch.norm(out[j] - z_src, p=1) for j in range(out.shape[0])])
        best_idx = torch.argmin(overlap_batch)
        if overlap_batch[best_idx] < best_overlap:
            best_overlap = overlap_batch[best_idx]
            # best_similarity = best_loss.item()
            x_src = tokens_batch[best_idx].unsqueeze(0)
        current_acts = sae.encode(h_batch[best_idx, -1, :]).top_acts
        # print(f"current: {torch.sort(current_acts, descending=True)}")
        # print(f"src: {torch.sort(top_acts_src, descending=True)}")
        best_loss = cos_sim(z_batch[best_idx], z_src) 
        best_loss = best_loss.item()
        
        print(f"Iteration {i+1} loss = {best_loss}")    
        print(f"Iteration {i+1} overlap_ratio={best_overlap}")   
        print(f"Iteration {i+1} input: {tokenizer.decode(x_src[0], skip_special_tokens=True)}")
        print("--------------------")
    print(f"Token {t} best loss: {best_loss}")
    print(f"Token {t} best overlap: {best_overlap}")

    losses.append(best_loss)
    overlaps.append(best_overlap.item())

attack_time = time.time() - attack_start_time
print(f"Total attack time: {attack_time:.2f} seconds")

print(f"All losses: {losses}")
print(f"All overlaps: {overlaps}")
print(f"Mean overlaps = {sum(overlaps) / len(overlaps)}")
