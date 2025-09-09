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
from utils import *
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/"
CACHE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", cache_dir=CACHE_DIR).to(DEVICE)
layer_num = 35
n_id = 66255
num_iters = 15
m = 400
batch_size = 100
log = True

model_type = "gemma2-9b"
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-9b-pt-res",
    filename="layer_35/width_131k/average_l0_94/params.npz",
    force_download=False,
    cache_dir = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/gemma2-9b-sae"
)
params = np.load(path_to_params)
sae = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

# ""
x1_raw_text = "The effect of lard and sunflower oil making part of a cirrhogenic ration with a high content of fat and deficient protein and choline on the level of total and esterified cholesterol and phospholipids in the blood serum and liver was studied."

if log:
    log_file_path = f"./results/neuronpedia-{model_type}/{layer_num}-94-{n_id}.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = open(log_file_path, "w")
x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
# x1 = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)

h1_raw_all = model(x1_raw, output_hidden_states=True).hidden_states[layer_num + 1][0]
for t_id in range(x1_raw.shape[-1]):
    h1_raw = h1_raw_all[t_id]
    z1_signed = h1_raw @ sae['W_enc'] + sae['b_enc']
    z1_raw, s1_raw, s1_raw_acts = extract_sae_features(h1_raw, sae, model_type)
    k = len(s1_raw)
    # initial_rank = (s1 == n_id).nonzero(as_tuple=True)[0].item()
    # print(initial_rank)
    if z1_raw[n_id] > 0:
        print(f"Target token index = {t_id}")
        print(f"Target token decoded: {tokenizer.decode(x1_raw[0, t_id:t_id+1], skip_special_tokens=True)}")
        
        best_rank = 0
        x1 = x1_raw.clone()
        r_id = random.randint(0, t_id)
        for i in range(num_iters):
            with torch.no_grad():
                embeddings = model.get_input_embeddings()(x1) 
            embeddings = embeddings.detach().clone().requires_grad_(True)
            h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[layer_num + 1][0][t_id]
            z1, s1, s1_acts = extract_sae_features(h1, sae, model_type, k)
            loss = -torch.nn.functional.log_softmax(z1, dim=0)[n_id]
            gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
            dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
            dot_prod[:, tokenizer.eos_token_id] = -float('inf')
            top_m_adv = (torch.topk(dot_prod, m).indices)[r_id]

            x1_batch = x1.repeat(batch_size, 1).clone()
            rand_top_m_idx = torch.randint(0, m, (batch_size,))
            x1_batch[:, r_id] = top_m_adv[rand_top_m_idx]

            with torch.no_grad():
                h1_batch = model(x1_batch, output_hidden_states=True).hidden_states[layer_num + 1]
                z1_batch, _, _ = extract_sae_features(h1_batch[:, t_id, :], sae, model_type, k)

            sorted_indices = torch.sort(z1_batch, dim=1, descending=True).indices
            rank_batch = (sorted_indices == n_id).nonzero(as_tuple=False)
            rank_per_sample = torch.full((z1_batch.shape[0],), z1_batch.shape[1], dtype=torch.long, device=z1_batch.device)
            rank_per_sample[rank_batch[:, 0]] = rank_batch[:, 1]
            best_idx = torch.argmax(rank_per_sample)
            x1 = x1_batch[best_idx].unsqueeze(0)
            current_rank = rank_per_sample[best_idx].item()
            if current_rank > best_rank:
                best_rank = current_rank
            current_loss = torch.nn.functional.log_softmax(z1_batch[best_idx])[n_id].item()
            x1_text = tokenizer.decode(x1[0][:x1.shape[-1]], skip_special_tokens=True)
            print(f"Token {t_id} Iteration {i+1} log likelihood = {current_loss}")
            print(f"Token {t_id} Iteration {i+1} best rank = {best_rank}")
            print(f"Token {t_id} Iteration {i+1} input text: {x1_text}")
            
            if z1_batch[best_idx][n_id] <= 0 and model_type == "gemma2-9b":
                print(f"Token {t_id} Neuron {n_id} successfully deactivated!")
                print("--------------------")
                break
if log:
    sys.stdout.close()
  