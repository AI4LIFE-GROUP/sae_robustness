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

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)

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
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_12/width_16k/average_l0_22/params.npz",
    force_download=False,
    cache_dir = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/gemma2-2b-sae"
)
params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
print({k:v.shape for k, v in pt_params.items()})

# def cos_sim(x, y):
#     # Normalize the tensors
#     norm_x = torch.norm(x)
#     norm_y = torch.norm(y)
    
#     # Compute the dot product
#     dot_product = torch.sum(x * y)
    
#     # Compute the cosine similarity
#     cosine_similarity = dot_product / (norm_x * norm_y)
    
#     return cosine_similarity

# def count_common(x, y):
#     num_common = 0
#     for elem in x:
#         if elem in y:
#             num_common += 1
#     return num_common

# attack_start_time = time.time()
# data_file = "./two_class_generated.csv"
# df = pd.read_csv(data_file)
# sample_idx = 78
# layer_num = 12
# num_top = 200

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
# x1_raw_text = df.iloc[sample_idx]['x1'][:-1]
# print(f"x1: {x1_raw_text}")

# x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
# x1_raw_processed = tokenizer(x1_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

# # model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
# # model.config.pad_token_id = model.config.eos_token_id

# h1_init = model(x1_raw_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
# z1_init = h1_init @ sae.W_enc + sae.b_enc
# sorted, indices = torch.sort(z1_init, descending=True)
# s1_init = indices[:num_top]
# s1_init_acts = sorted[:num_top]

# num_iters = 50
# k = 200
# num_adv = 1
# batch_size = 100

# model.to(DEVICE)
# model.eval()
# # best_loss = float("Inf")
# best_overlap = 1.0
# losses = []
# overlaps = []
# x1_init = model.generate(
#     tokenizer(df.iloc[sample_idx]['x1'][:-1], return_tensors="pt")['input_ids'].to(DEVICE),
#     max_length=x1_raw.shape[-1] + num_adv,  # Maximum length of the generated text
#     do_sample=False,
#     temperature=None,
#     top_p=None,
#     num_return_sequences=1,  # Number of sequences to generate
#     no_repeat_ngram_size=2,  # To avoid repeating the same n-grams
#     # early_stopping=True,  # Stop generating when it seems complete
# )

# # print(f"x1_initial shape={x1.shape}")
# x1_init_text = tokenizer.decode(x1_init[0], skip_special_tokens=True)
# x1_init_processed = tokenizer(x1_init_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
# # print(f"x1 init processed: {x1_init_processed}", x1_init_processed.shape)
# h1 = model(x1_init_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
# z1 = h1 @ sae.W_enc + sae.b_enc
# sorted, indices = torch.sort(z1, descending=True)
# s1 = indices[:num_top]
# s1_acts = sorted[:num_top]
# # print(s1, s1_init)
# print(f"Initial overlap = {count_common(s1, s1_init) / len(s1_init)}")

# x1 = x1_init_processed
# best_x1 = x1_init_text
# for i in range(num_iters):
#     with torch.no_grad():
#         out = model(x1, output_hidden_states=True)
    
#     embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
    
#     # Forward pass again with embeddings
#     h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
#     z1 = h1 @ sae.W_enc + sae.b_enc
    
#     # Compute similarity loss
#     loss = -cos_sim(z1, z1_init)

#     # Calculate gradients for adversarial attack
#     gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
#     dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)

#     # Remove influence of special tokens
#     cls_token_idx = tokenizer.encode('[CLS]')[1]
#     sep_token_idx = tokenizer.encode('[SEP]')[1]
#     dot_prod[:, cls_token_idx] = -float('inf')
#     dot_prod[:, sep_token_idx] = -float('inf')

#     # Get top k adversarial tokens (suffix mode)
#     top_k_adv = (torch.topk(dot_prod, k).indices)[x1_init.shape[-1] - num_adv:x1_init.shape[-1]]

#     # Method 1: Random sampling
#     x1_batch = x1.repeat(batch_size, 1).clone()
#     random_idx = torch.randint(0, num_adv, (batch_size,))
#     random_top_k_idx = torch.randint(0, k, (batch_size,))
#     batch_indices = torch.arange(batch_size)
#     x1_batch[:, x1_init.shape[-1] - num_adv:x1_init.shape[-1]][batch_indices, random_idx] = top_k_adv[random_idx, random_top_k_idx]

#     # Method 2: Exhaustive; sometimes stuck between two sequences
#     # x1_batch = x1.repeat(num_adv*k, 1).clone()
#     # for d1 in torch.arange(num_adv):
#     #     for d2 in torch.arange(k):
#     #         x1_batch[d1*k + d2, d1-num_adv] = top_k_adv[d1, d2]

#     with torch.no_grad():
#         new_embeds = model(x1_batch, output_hidden_states=True).hidden_states[0]
#         h1_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num + 1]
#         z1_batch = h1_batch[:, -1, :] @ sae.W_enc + sae.b_enc
#         # z1_batch = sae.pre_acts(h1_batch[:, -1, :])
#     sorted, indices = torch.sort(z1_batch, dim=-1, descending=True)
#     s1_batch = indices[:, :num_top]
#     # s1_acts_batch = sorted[:, :num_top]
#     # s1_batch = sae.encode(h1_batch[:, -1, :]).top_indices # (k, num_topk=192)
#     overlap_batch = torch.tensor([count_common(s1_batch[j], s1_init) / len(s1_init) for j in range(s1_batch.shape[0])])
    
#     best_idx = torch.argmin(overlap_batch)
#     x1 = x1_batch[best_idx].unsqueeze(0)
#     # print(f"best idx = {best_idx}")
#     if overlap_batch[best_idx] < best_overlap:
#         best_overlap = overlap_batch[best_idx]
#         best_x1 = x1[0][:x1_init.shape[-1]]
#     best_loss = cos_sim(z1_batch[best_idx], z1_init).item()
#     x1_text = tokenizer.decode(best_x1, skip_special_tokens=True)
#     losses.append(best_loss)
#     overlaps.append(best_overlap.item())

#     print(f"Iteration {i+1} loss = {best_loss}")
#     print(f"Iteration {i+1} best overlap ratio = {best_overlap}")
#     print(f"Iteration {i+1} input text: {x1_text}")
#     print("--------------------")

# print(losses)
# print(overlaps)
# print(f"Best loss = {best_loss}")
# print(f"Best overlap = {best_overlap}")
# print(f"Best x1 = {tokenizer.decode(best_x1, skip_special_tokens=True)}")