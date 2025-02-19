import sys
sys.path.append('../sae')
from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

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
        if elem in y:
            num_common += 1
    return num_common

attack_start_time = time.time()
data_file = "./sae_samples_50.csv"
df = pd.read_csv(data_file)
sample_idx = 15
layer_num = 25
sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
src_text = df.iloc[sample_idx]['x1']
target_text = df.iloc[sample_idx]['x2']
print(f"x1: {src_text}")
print(f"x2: {target_text}")

x_src = tokenizer(src_text, return_tensors="pt")['input_ids'].to(DEVICE)
x_target = tokenizer(target_text, return_tensors="pt")['input_ids'].to(DEVICE)

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
model.config.pad_token_id = model.config.eos_token_id

h_src = model(x_src, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z_src = sae.pre_acts(h_src)
top_idx_src = sae.encode(h_src).top_indices
top_acts_src = sae.encode(h_src).top_acts

h_target = model(x_target, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z_target = sae.pre_acts(h_target)
top_idx_target = sae.encode(h_target).top_indices
top_acts_target = sae.encode(h_target).top_acts

print(f"Initial overlap = {count_common(top_idx_src, top_idx_target) / len(top_idx_target)}")

num_iters = 100
k = 500
num_adv = 5
batch_size = 1400

model.to(DEVICE)
model.eval()
# best_loss = float("Inf")
best_overlap = 0.0
losses = []
overlaps = []
# print(f"x_src shape={x_src.shape}")
x1 = model.generate(
    x_src,
    max_length=x_src.shape[-1] + num_adv,  # Maximum length of the generated text
    do_sample=False,
    temperature=None,
    top_p=None,
    num_return_sequences=1,  # Number of sequences to generate
    no_repeat_ngram_size=2,  # To avoid repeating the same n-grams
    # early_stopping=True,  # Stop generating when it seems complete
)
# print(f"x1_initial shape={x1.shape}")
x1_text = tokenizer.decode(x1[0], skip_special_tokens=True)
best_x1 = x1_text
print(f"x1 initial: {x1_text}")

h1 = model(x1, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z1 = sae.pre_acts(h1)
top_idx_1 = sae.encode(h1).top_indices
top_acts_1 = sae.encode(h1).top_acts
print(f"x1 initial overlap = {count_common(top_idx_1, top_idx_target) / len(top_idx_target)}")

for i in range(num_iters):
    with torch.no_grad():
        out = model(x1, output_hidden_states=True)
    
    embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
    
    # Forward pass again with embeddings
    h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
    z1 = sae.pre_acts(h1)
    
    # Compute similarity loss
    # loss = cos_sim(h1, h_target)
    loss = -torch.norm(h1 - h_target, p=2)

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

    # Method 1: Random sampling
    x1_batch = x1.repeat(batch_size, 1).clone()
    random_idx = torch.randint(0, num_adv, (batch_size,))
    random_top_k_idx = torch.randint(0, k, (batch_size,))
    batch_indices = torch.arange(batch_size)
    x1_batch[:, -num_adv:][batch_indices, random_idx] = top_k_adv[random_idx, random_top_k_idx]

    # Method 2: Exhaustive; sometimes stuck between two sequences
    # x1_batch = x1.repeat(num_adv*k, 1).clone()
    # for d1 in torch.arange(num_adv):
    #     for d2 in torch.arange(k):
    #         x1_batch[d1*k + d2, d1-num_adv] = top_k_adv[d1, d2]

    with torch.no_grad():
        new_embeds = model(x1_batch, output_hidden_states=True).hidden_states[0]
        h1_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num + 1]
        z1_batch = sae.pre_acts(h1_batch[:, -1, :])

    top_idx_batch = sae.encode(h1_batch[:, -1, :]).top_indices # (k, num_topk=192)
    overlap_batch = torch.tensor([count_common(top_idx_batch[j], top_idx_target) / len(top_idx_target) for j in range(top_idx_batch.shape[0])])
    
    best_idx = torch.argmax(overlap_batch)
    x1 = x1_batch[best_idx].unsqueeze(0)
    if overlap_batch[best_idx] > best_overlap:
        best_overlap = overlap_batch[best_idx]
        best_x1 = x1
    # best_loss = cos_sim(h1_batch[best_idx], h_target).item()
    best_loss = torch.norm(h1_batch[best_idx] - h_target, p=2).item()
    x1_text = tokenizer.decode(x1[0], skip_special_tokens=True)
    losses.append(best_loss)
    overlaps.append(best_overlap.item())

    print(f"Iteration {i+1} loss = {best_loss}")
    print(f"Iteration {i+1} best overlap ratio = {best_overlap}")
    print(f"Iteration {i+1} input text: {x1_text}")
    print("--------------------")

print(losses)
print(overlaps)
print(f"Best loss = {best_loss}")
print(f"Best overlap = {best_overlap}")
print(f"Best x1 = {tokenizer.decode(best_x1[0], skip_special_tokens=True)}")