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
data_file = "./two_class_generated.csv"
df = pd.read_csv(data_file)
sample_idx = 24
layer_num = 10
sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
x1_raw_text = df.iloc[sample_idx]['x1'][:-1]
print(f"x1: {x1_raw_text}")

x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
x1_raw_processed = tokenizer(x1_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
model.config.pad_token_id = model.config.eos_token_id

h1_init = model(x1_raw_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z1_init = sae.pre_acts(h1_init)
s1_init = sae.encode(h1_init).top_indices

num_iters = 10
k = 200
batch_size = 50

model.to(DEVICE)
model.eval()

x1_init_processed = tokenizer(x1_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
# print(f"x1 init processed: {x1_init_processed}", x1_init_processed.shape)
h1 = model(x1_init_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z1 = sae.pre_acts(h1)
s1 = sae.encode(h1).top_indices
s1_acts = sae.encode(h1).top_acts

losses = []
overlaps = []

for t in range(1, x1_raw.shape[-1]):
    x1 = x1_init_processed.clone()
    best_loss = float('inf')
    best_overlap = 1.0

    for i in range(num_iters):
        with torch.no_grad():
            out = model(x1, output_hidden_states=True)
        embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
        h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
        z1 = sae.pre_acts(h1)
        # drift = torch.norm(sae_out - z_src, p=1)
        cosine_loss = cos_sim(z1, z1_init)
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

        # Get top k adversarial tokens (replacement mode)
        top_k_adv = (torch.topk(dot_prod, k).indices)[t] 

        x1_batch = x1.repeat(k, 1)  # shape (k, seq_len)
        # Replace the t-th token in each row with top_k_adv
        x1_batch[:, t] = top_k_adv
        # print(f"tokens batch {x1_batch.shape}")

        with torch.no_grad():
            new_embeds = model(x1_batch, output_hidden_states=True).hidden_states[0]
            h1_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num + 1] # (k, seq_len, lm_dim)
            z1_batch = sae.pre_acts(h1_batch[:, -1, :]) # (k, sae_dim)
        s1_batch = sae.encode(h1_batch[:, -1, :]).top_indices # (k, num_topk=192)
        overlap_batch = torch.tensor([count_common(s1_batch[j], s1_init) / len(s1_init) for j in range(s1_batch.shape[0])])
        # loss_batch = torch.tensor([cos_sim(out[j], z_src) - 0.005 * torch.norm(out[j] - z_src, p=1) for j in range(out.shape[0])])
        best_idx = torch.argmin(overlap_batch)
        if overlap_batch[best_idx] < best_overlap:
            best_overlap = overlap_batch[best_idx]
            # best_similarity = best_loss.item()
        x1 = x1_batch[best_idx].unsqueeze(0)
        current_acts = sae.encode(h1_batch[best_idx, -1, :]).top_acts
        # print(f"current: {torch.sort(current_acts, descending=True)}")
        # print(f"src: {torch.sort(top_acts_src, descending=True)}")
        best_loss = cos_sim(z1_batch[best_idx], z1_init) 
        best_loss = best_loss.item()
        
        print(f"Iteration {i+1} loss = {best_loss}")    
        print(f"Iteration {i+1} overlap_ratio={best_overlap}")   
        print(f"Iteration {i+1} input: {tokenizer.decode(x1[0][:x1_raw.shape[-1]], skip_special_tokens=True)}")
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
