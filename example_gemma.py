import sys
import os
sys.path.append('../sae')
from sae import Sae
os.environ["HF_HOME"] = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/huggingface"
os.environ["HF_HUB_CACHE"] = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/huggingface/hub"
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import hf_hub_download

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def jump_relu(x, theta):
    return x * (x > theta).float()

def count_common(x, y):
    num_common = 0
    for elem in x:
        if elem in y:
            num_common += 1
    return num_common
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)

# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release="gemma-2b-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
#     sae_id="blocks.12.hook_resid_post",  
#     device=DEVICE
# )
# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release="gemma-9b-res-pt-res",  # see other options in sae_lens/pretrained_saes.yaml
#     sae_id="layer_20/width_131k/average_l0_11",  
#     device=DEVICE
# )
layer_num = 30
num_top = 50

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-9b-pt-res",
    filename="layer_30/width_131k/average_l0_170/params.npz",
    force_download=False,
    cache_dir = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/gemma2-9b-sae"
)
params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
W_enc = pt_params['W_enc']
b_enc = pt_params['b_enc']
theta = pt_params['threshold']

x1_text = "input text: The film explores love and trauma through non-linear storytelling, blending magical realism with emotionally raw performances.\n\n examined"
x2_text = "Encryption secures sensitive digital communication by converting readable data into unreadable ciphertext"

x1 = tokenizer(x1_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
x2 = tokenizer(x2_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

h1 = model(x1, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
    
z1 = jump_relu(h1 @ W_enc + b_enc, theta)
sorted, indices = torch.sort(z1, descending=True)
s1 = indices[:num_top]
s1_acts = sorted[:num_top]
print(f"s1: {indices[:50]}")
print(f"s1_acts: {sorted[:50]}")
# s1_raw = torch.nonzero(z1_raw > 0, as_tuple=True)[0]
# s1_raw_acts = z1_raw[s1_raw]

h2 = model(x2, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z2 = jump_relu(h2 @ W_enc + b_enc, theta)
sorted, indices = torch.sort(z2, descending=True)
print(f"s2: {indices[:50]}")
print(f"s2_acts: {sorted[:50]}")
s2 = indices[:num_top]
s2_acts = sorted[:num_top]
# s2 = torch.nonzero(z2 > 0, as_tuple=True)[0]
# s2_acts = z2[s2]
overlap = count_common(s1, s2) / len(s2)
print(f"overlap = {overlap}")