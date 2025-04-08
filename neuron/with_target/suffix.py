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

data_file = "./two_class_generated.csv"
df = pd.read_csv(data_file)
sample_idx = 20
layer_num = 25
activate = True
num_selected = 10
sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
x1_raw_text = df.iloc[sample_idx]['x1'][:-1]
x2_raw_text = df.iloc[sample_idx]['x2'][:-1]
print(f"x1: {x1_raw_text}")
print(f"x2: {x2_raw_text}")

x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
x2_raw = tokenizer(x2_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
x1_raw_processed = tokenizer(x1_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
x2_raw_processed = tokenizer(x2_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
model.config.pad_token_id = model.config.eos_token_id

h1_raw = model(x1_raw_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z1_raw = sae.pre_acts(h1_raw)
s1_raw = sae.encode(h1_raw).top_indices
s1_raw_acts = sae.encode(h1_raw).top_acts

h2 = model(x2_raw_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z2 = sae.pre_acts(h2)
s2 = sae.encode(h2).top_indices
s2_acts = sae.encode(h2).top_acts

print(f"Initial overlap = {count_common(s1_raw, s2) / len(s2)}")

num_iters = 10
k = 300
num_adv = 1
batch_size = 100

model.to(DEVICE)
model.eval()
# best_loss = float("Inf")
# print(f"x_src shape={x_src.shape}")
x1_init = model.generate(
    tokenizer(df.iloc[sample_idx]['x1'][:-1], return_tensors="pt")['input_ids'].to(DEVICE),
    max_length=x1_raw.shape[-1] + num_adv,  # Maximum length of the generated text
    do_sample=False,
    temperature=None,
    top_p=None,
    num_return_sequences=1,  # Number of sequences to generate
    no_repeat_ngram_size=2,  # To avoid repeating the same n-grams
    # early_stopping=True,  # Stop generating when it seems complete
)
# print(f"x1_initial shape={x1.shape}")
x1_init_text = tokenizer.decode(x1_init[0], skip_special_tokens=True)

# print(f"x1 raw text: {x1_raw_text}")
# print(f"x1 raw: {x1_raw}", x1_raw.shape)
# print(f"x1 init text: {x1_init_text}")
# print(f"x1 init: {x1_init}", x1_init.shape)
x1_init_processed = tokenizer(x1_init_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
# print(f"x1 init processed: {x1_init_processed}", x1_init_processed.shape)
h1 = model(x1_init_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
z1 = sae.pre_acts(h1)
s1 = sae.encode(h1).top_indices
s1_acts = sae.encode(h1).top_acts
# print(f"x1 initial overlap = {count_common(top_idx_1, top_idx_target) / len(top_idx_target)}")
# if activate:
#     neuron_list = s2[~torch.isin(s2, s1)]
# else:
#     neuron_list = s1[~torch.isin(s1, s2)]

if activate:
    # Neurons in s2 but not in s1 
    mask = ~torch.isin(s2, s1)
    # s2_neurons = s2[mask]
    # new_activations = s2_acts[mask]
    top_indices = torch.topk(s2_acts[mask], k=min(num_selected, s2_acts[mask].numel())).indices  # highest in s2 - s1
    neuron_list = s2[mask][top_indices]
else:
    # Neurons in s1 but not in s2 
    mask = ~torch.isin(s1, s2)
    top_indices = torch.topk(s1_acts[mask], k=min(num_selected, s1_acts[mask].numel())).indices  # highest in s1 - s2
    neuron_list = s1[mask][top_indices]

all_final_ranks = []
success_count = 0
count = 0
for n_id in neuron_list:
    x1 = x1_init_processed.clone()
    if activate:
        best_rank = z1.shape[-1]
    else:
        best_rank = 0
    losses = []
    ranks = []
    for i in range(num_iters):
        with torch.no_grad():
            out = model(x1, output_hidden_states=True)
        
        embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
        
        # Forward pass again with embeddings
        h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
        z1 = sae.pre_acts(h1)
        
        # Compute similarity loss
        # loss = cos_sim(z1, z_target)
        if activate:
            loss = torch.nn.functional.log_softmax(z1, dim=0)[n_id]
        else:
            loss = -torch.nn.functional.log_softmax(z1, dim=0)[n_id]

        # Calculate gradients for adversarial attack
        gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
        dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)

        # Remove influence of special tokens
        cls_token_idx = tokenizer.encode('[CLS]')[1]
        sep_token_idx = tokenizer.encode('[SEP]')[1]
        dot_prod[:, cls_token_idx] = -float('inf')
        dot_prod[:, sep_token_idx] = -float('inf')

        # Get top k adversarial tokens (suffix mode)
        top_k_adv = (torch.topk(dot_prod, k).indices)[x1_init.shape[-1] - num_adv:x1_init.shape[-1]]

        # Method 1: Random sampling
        x1_batch = x1.repeat(batch_size, 1).clone()
        random_idx = torch.randint(0, num_adv, (batch_size,))
        random_top_k_idx = torch.randint(0, k, (batch_size,))
        batch_indices = torch.arange(batch_size)
        x1_batch[:, x1_init.shape[-1] - num_adv:x1_init.shape[-1]][batch_indices, random_idx] = top_k_adv[random_idx, random_top_k_idx]

        with torch.no_grad():
            new_embeds = model(x1_batch, output_hidden_states=True).hidden_states[0]
            h1_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num + 1]
            z1_batch = sae.pre_acts(h1_batch[:, -1, :])

        s1_batch = sae.encode(h1_batch[:, -1, :]).top_indices # (k, num_topk=192)
        # overlap_batch = torch.tensor([count_common(top_idx_batch[j], top_idx_target) / len(top_idx_target) for j in range(top_idx_batch.shape[0])])
        
        # Get activations of the target neuron
        neuron_acts = z1_batch[:, n_id]  # shape: (batch_size,)

        # Compute rank of the target neuron in each sample (lower rank = higher activation)
        # Sort in descending order to rank by highest activation first
        sorted_vals, sorted_indices = torch.sort(z1_batch, dim=1, descending=True)

        # Get rank of neuron_id for each sample in batch
        rank_batch = (sorted_indices == n_id).nonzero(as_tuple=False)
        # Each row of rank_batch is [sample_idx, rank]; we want rank per sample
        rank_per_sample = torch.full((z1_batch.shape[0],), z1_batch.shape[1], dtype=torch.long, device=z1_batch.device)
        rank_per_sample[rank_batch[:, 0]] = rank_batch[:, 1]  # (batch_size,)
        # print(rank_per_sample)
        
        if activate:
            # Lower is better (rank 0 = highest)
            best_idx = torch.argmin(rank_per_sample)
        else:
            best_idx = torch.argmax(rank_per_sample)

        x1 = x1_batch[best_idx].unsqueeze(0)
        if (activate and rank_per_sample[best_idx] < best_rank) or (not activate and rank_per_sample[best_idx] > best_rank):
            best_rank = rank_per_sample[best_idx]
        best_loss = torch.nn.functional.log_softmax(z1_batch[best_idx])[n_id].item()
        x1_text = tokenizer.decode(x1[0][:x1_init.shape[-1]], skip_special_tokens=True)
        losses.append(best_loss)
        ranks.append(best_rank.item())

        print(f"Iteration {i+1} loss = {best_loss}")
        print(f"Iteration {i+1} best rank = {best_rank}")
        print(f"Iteration {i+1} input text: {x1_text}")
        if activate and best_rank < len(s2):
            print(f"Neuron {n_id} successfully activated!")
            print("--------------------")
            success_count += 1
            break
        if not activate and best_rank > len(s2):
            print(f"Neuron {n_id} successfully deactivated!")
            print("--------------------")
            success_count += 1
            break
    count += 1
    if activate and best_rank > len(s2):
        print(f"Neuron {n_id} cannot be activated")
        print("--------------------")
    if not activate and best_rank < len(s2):
        print(f"Neuron {n_id} cannot be deactivated")
        print("--------------------")
    all_final_ranks.append(best_rank.item())
    print(f"{success_count} out of {count} attacks are successful!")

print(f"Successful rate = {success_count / len(neuron_list)}")
print(f"All final ranks = {all_final_ranks}")

