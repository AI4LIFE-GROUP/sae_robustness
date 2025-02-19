from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

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

data_file = "./sae_samples_50.csv"
df = pd.read_csv(data_file)
sample_idx = 10
layer_num = 20
sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# src_text = "The cat slept peacefully on the sunny windowsill "
# target_text = "An astronaut floated weightlessly in the vast expanse of space "
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

x_src = torch.tensor(tokenizer.encode(src_text)).unsqueeze(0).to(DEVICE)
x_src_old = x_src.clone()

model.to(DEVICE)

overlaps = []
overlap_increases = []
overlap_increase_ratios = []

# num_sign_agreements = []
print(f"Original Input: {tokenizer.decode(x_src[0], skip_special_tokens=True)}")
# print(x_src)
for t in range(1, x_src.shape[-1]):
    best_overlap = 0.0
    best_x = x_src.clone()

    # Iterate through all tokens
    vocab = tokenizer.get_vocab()
    count = 0
    for i, (token, token_id) in tqdm(enumerate(vocab.items())):
        # print(f"Token: {token}, ID: {token_id}")
        if not re.fullmatch(r"[a-zA-Z]+", token):
            continue
        adv_prompt_tokens = x_src.clone().detach()
        adv_prompt_tokens[0, t] = token_id

        with torch.no_grad():
            lm_out = model(adv_prompt_tokens, output_hidden_states=True)
            lm_h = lm_out.hidden_states[layer_num+1][0][-1] # (,hidden_dim)
            top_idx_src = sae.encode(lm_h).top_indices
            num_overlap = count_common(top_idx_src, top_idx_target)
            overlap = num_overlap / len(top_idx_target)
        if overlap > best_overlap:
            best_overlap = overlap
            best_x = adv_prompt_tokens

    print(f"Token {t} best input: {tokenizer.decode(best_x[0], skip_special_tokens=True)}")
    print(f"Token {t} best overlap: {best_overlap}")
        
    overlap_increase = best_overlap - initial_overlap
    overlap_increase_ratio = overlap_increase / initial_overlap
    print(f"Token {t} overlap increase: {overlap_increase}, {overlap_increase_ratio}")
    
    overlaps.append(overlap)
    overlap_increases.append(overlap_increase)
    overlap_increase_ratios.append(overlap_increase_ratio)

print(f"All overlaps: {overlaps}")
print(f"All overlap increases: {overlap_increases}")
print(f"All overlap increase ratios: {overlap_increase_ratios}")

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].plot(np.arange(1, num_iters+1), np.array(similarities))
# axs[0].set_xlabel('Iteration')
# axs[0].set_ylabel('Similarity')
# axs[0].set_title('Cosine Similarity (Raw Activations) vs. Iteration')

# axs[1].plot(np.arange(1, num_iters+1), np.array(overlaps))
# axs[1].set_xlabel('Iteration')
# axs[1].set_ylabel('Neuron Overlap')
# axs[1].set_title('Neuron Overlap vs. Iteration')

# axs[2].plot(np.arange(1, num_iters+1), num_sign_agreements)
# axs[2].set_xlabel('Iteration')
# axs[2].set_ylabel('Sign Agreements')
# axs[2].set_title('Sign Agreements vs. Iteration')
# plt.savefig("./results/llama3-8b/layer-20/500_5_1000-prefix-2.png")
# plt.show()

        