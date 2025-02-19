import sys
sys.path.append('../sae')
from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

num_iters = 7
k = 200
# batch_size = 1000

x_src = torch.tensor(tokenizer.encode(src_text)).unsqueeze(0).to(DEVICE)
x_src_old = x_src.clone()

model.to(DEVICE)
best_loss = 100.0

losses = []
overlaps = []

print(f"Original Input: {tokenizer.decode(x_src[0], skip_special_tokens=True)}")
for t in range(1, x_src.shape[-1]):
    x_src = x_src_old
    best_loss = 100.0
    best_overlap = 1.0

    for i in range(num_iters):
        with torch.no_grad():
            out = model(x_src, output_hidden_states=True)
        embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
        lm_out = model(inputs_embeds=embeddings, output_hidden_states=True)
        sae_out = sae.pre_acts(lm_out.hidden_states[layer_num + 1][0][-1])
        
        # --- Vectorized Random Token Selection ---
        vocab_list = torch.tensor(list(tokenizer.get_vocab().values()), device=DEVICE)
        random_tokens = vocab_list[torch.randint(0, len(vocab_list), (k,), device=DEVICE)]

        # Clone input and replace `t`-th token for all k samples
        tokens_batch = x_src.repeat(k, 1)
        tokens_batch[:, t] = random_tokens

        # tokens_batch = []

        # for k_idx in range(k):
        #     batch_item = x_src.clone().detach()
        #     vocab_list = list(tokenizer.get_vocab().values())  # Get list of all token IDs
        #     random_token = np.random.choice(vocab_list)  # Randomly select one token ID
        #     batch_item[0, t] = torch.tensor(random_token, device=batch_item.device)  # Assign to token position
        #     tokens_batch.append(batch_item)

        # tokens_batch = torch.cat(tokens_batch, dim=0)

        with torch.no_grad():
            new_embeds = model(tokens_batch, output_hidden_states=True).hidden_states[0]
            h_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num + 1] # (k, seq_len, lm_dim)
            z_batch = sae.pre_acts(h_batch[:, -1, :]) # (k, sae_dim)
        # loss_batch = torch.tensor([cos_sim(out[j], z_src) for j in range(out.shape[0])])
        top_idx_batch = sae.encode(h_batch[:, -1, :]).top_indices # (k, num_topk=192)
        overlap_batch = torch.tensor([count_common(top_idx_batch[j], top_idx_src) / len(top_idx_src) for j in range(top_idx_batch.shape[0])])
        
        best_idx = torch.argmin(overlap_batch)
        if overlap_batch[best_idx] < best_overlap:
            best_overlap = overlap_batch[best_idx]
            x_src = tokens_batch[best_idx].unsqueeze(0)

        best_loss = cos_sim(z_batch[best_idx], z_src) 
        best_loss = best_loss.item()
        # overlaps.append(num_overlap)
        # new_acts = sae.pre_acts(out.hidden_states[layer_num + 1][0][-1])
        # num_agrees = torch.sum(torch.sign(new_acts == latent_acts_target))
        # num_sign_agreements.append(num_agrees.item())
        print(f"Iteration {i+1} loss = {best_loss}")    
        print(f"Iteration {i+1} overlap_ratio={best_overlap}")   
        # print(f"Iteration {i+1} num_sign_agreements = {num_agrees} out of {new_acts.shape[0]}")  
        print(f"Iteration {i+1} input: {tokenizer.decode(x_src[0], skip_special_tokens=True)}")
        print("--------------------")
    print(f"Token {t} best similarity: {best_loss}")
    print(f"Token {t} best overlap: {best_overlap}")
    
    losses.append(best_loss)
    overlaps.append(best_overlap.item())

attack_time = time.time() - attack_start_time
print(f"Total attack time: {attack_time:.2f} seconds")
print(f"All losses: {losses}")
print(f"All overlaps: {overlaps}")
print(f"Mean overlaps = {sum(overlaps) / len(overlaps)}")
