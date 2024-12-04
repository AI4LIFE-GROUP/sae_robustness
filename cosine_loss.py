from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/sae/"
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
        if elem in y:
            num_common += 1
    return num_common

layer_num = 25
sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
src_text = "The cat slept peacefully on the sunny windowsill "
target_text = "An astronaut floated weightlessly in the vast expanse of space "
x_src = tokenizer(src_text, return_tensors="pt").to(DEVICE)
x_target = tokenizer(target_text, return_tensors="pt").to(DEVICE)

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)

# Extract initial hidden states
h_src = model(**x_src, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
h_target = model(**x_target, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]

# SAE encoding
z_src = sae.pre_acts(h_src)
top_idx_src = sae.encode(h_src).top_indices
z_target = sae.pre_acts(h_target)
top_idx_target = sae.encode(h_target).top_indices
print(f"Initial overlap = {count_common(top_idx_src, top_idx_target) / len(top_idx_target)}")

num_iters = 80
k = 500
num_adv = 5
num_candidates = 1
batch_size = 1000
mode = 'suffix'

model.to(DEVICE)
model.eval()
best_loss = float("Inf")
best_overlap_ratio = 0.0
similarities = []
overlaps = []
print(f"Original Input: {src_text}")

# Setup the modified input if suffix mode is used
if mode == 'suffix':
    x_src_text = src_text + "* " * num_adv

x_src = tokenizer(x_src_text, return_tensors="pt").to(DEVICE)['input_ids']

for i in range(num_iters):
    with torch.no_grad():
        out = model(x_src, output_hidden_states=True)
    
    embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
    
    # Forward pass again with embeddings
    h_src = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
    z_src = sae.pre_acts(h_src)
    
    # Compute similarity loss
    loss = -cos_sim(z_src, z_target)

    # Calculate gradients for adversarial attack
    gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
    dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)

    # Remove influence of special tokens
    cls_token_idx = tokenizer.encode('[CLS]')[1]
    sep_token_idx = tokenizer.encode('[SEP]')[1]
    dot_prod[:, cls_token_idx] = -float('inf')
    dot_prod[:, sep_token_idx] = -float('inf')

    # Get top k adversarial tokens (suffix mode)
    top_k_adv = (torch.topk(dot_prod, k).indices)[-num_adv:]

    x_batch = []
    use_random = i < int(0.5 * num_iters)
    for _ in range(batch_size):
        random_idx = torch.randint(0, num_adv, (1,))
        random_top_k_idx = torch.randint(0, k, (1,))
        batch_item = x_src.clone().detach()

        if mode == 'suffix':
            batch_item[0, -num_adv:][random_idx] = top_k_adv[random_idx, random_top_k_idx]

        x_batch.append(batch_item)

    x_batch = torch.cat(x_batch, dim=0)

    with torch.no_grad():
        new_embeds = model(x_batch, output_hidden_states=True).hidden_states[0]
        h_src_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num + 1]
        h_src_batch = sae.pre_acts(h_src_batch[:, -1, :])

    # Compute losses for batch
    loss_batch = torch.tensor([-cos_sim(h_src_batch[j], z_target) for j in range(h_src_batch.shape[0])])
    best_indices = torch.argsort(loss_batch)[:num_candidates]
    candidate_idx = torch.randint(0, num_candidates, (1,))
    best_idx = best_indices[candidate_idx].item()

    # Update best_loss and x_src conditionally
    if loss_batch[best_idx] < best_loss:
        best_loss = loss_batch[best_idx]
        x_src = x_batch[best_idx].unsqueeze(0)

    x_src_decoded = tokenizer.batch_decode(x_src, skip_special_tokens=True)[0]
    similarities.append(-loss_batch[best_idx].item())

    with torch.no_grad():
        h_src_new = model(x_batch[best_idx].unsqueeze(0), output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
    
    top_idx_src_new = sae.encode(h_src_new).top_indices
    num_overlaps = count_common(top_idx_src_new, top_idx_target)
    overlap_ratio = num_overlaps / len(top_idx_src_new)
    if overlap_ratio > best_overlap_ratio:
        best_overlap_ratio = overlap_ratio
    overlaps.append(overlap_ratio)

    print(f"Iteration {i+1} loss = {loss_batch[best_idx].item()}")
    print(f"Iteration {i+1} overlap ratio = {overlap_ratio}")
    print(f"Iteration {i+1} input text: {x_src_decoded}")
    print("--------------------")

print(similarities)
print(overlaps)
print(f"Best loss = {best_loss.item()}, Best overlap ratio = {best_overlap_ratio}")
