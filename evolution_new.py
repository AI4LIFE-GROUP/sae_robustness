from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

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

# Simple function to count the number of shared elements in two lists
def count_common(x, y):
    num_common = 0
    for elem in x:
        if elem in y:
            num_common += 1
    return num_common

def greedy_coordinate_gradient(x, L, I, T, model, layer_num, sae, z_target, tokenizer, k=50, B=50):
    """
    Greedy Coordinate Gradient (GCG) algorithm for mutation.
    """
    # Get z2 sparse representatioh
    z2_idx = sae.select_topk(z_target).top_indices
    for _ in range(T):
        with torch.no_grad():
            embeddings = model(x, output_hidden_states=True).hidden_states[0].clone().detach().requires_grad_(True)
        lm_out = model(inputs_embeds=embeddings, output_hidden_states=True)

        # Get z1 before TopK
        sae_out = sae.pre_acts(lm_out.hidden_states[layer_num + 1][0][-1])

        # Continuous loss used to match z1 and z2
        loss = -cos_sim(sae_out, z_target)

        # The gradients should be with respect to the token embeddings
        gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
        
        # Get the gradients for all tokens in vocabulary
        dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
        
        # Make sure special tokens are not selected
        cls_token_idx = tokenizer.encode('[CLS]')[1]
        sep_token_idx = tokenizer.encode('[SEP]')[1]
        dot_prod[:, cls_token_idx] = -float('inf')
        dot_prod[:, sep_token_idx] = -float('inf')

        # Select top k promising tokens
        top_k_adv = torch.topk(dot_prod, k).indices # shape: (x_len, k)

        batch_candidates = []
        for _ in range(B):
            new_tokens = x.clone()
            idx_1 = random.choice(list(I)) # Index in suffix
            idx_2 = random.choice(range(k)) # Index in k most promising tokens
            new_tokens[:, idx_1] = top_k_adv[idx_1, idx_2]
            batch_candidates.append(new_tokens)

        # Evaluate all candidates based on neuron overlap ratio and choose the best
        best_idx = -1
        z1_old_idx = sae.encode(model(x, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]).top_indices
        best_overlap = count_common(z1_old_idx, z2_idx) / len(z1_old_idx)
        for idx, candidate in enumerate(batch_candidates):
            with torch.no_grad():
                z1_idx = sae.encode(model(candidate, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]).top_indices
                overlap = count_common(z1_idx, z2_idx) / len(z1_idx)
            if overlap > best_overlap: # Update with the best candidate
                best_idx = idx
                best_overlap = overlap
        if best_idx != -1:
            x = batch_candidates[best_idx] 

    return x # Best sequence after GCG, of shape (1, x1_len+suffix_len)


def evolutionary_algorithm(x1, x2, L, model, layer_num, sae, tokenizer, N=100, G=30, k=50, B=50, T=30):
    """
    Evolutionary algorithm to optimize adversarial suffix.
    """
    x1_text = tokenizer.decode(x1["input_ids"][0], skip_special_tokens=True)
    print(f"x1 text: {x1_text}")
    z_target = sae.pre_acts(model(x2["input_ids"], output_hidden_states=True).hidden_states[layer_num + 1][0][-1])
    z2_idx = sae.select_topk(z_target).top_indices
    x1_len = x1["input_ids"].size(1)
    I = list(range(x1_len, x1["input_ids"].size(1) + L))

    # Generate initial population (each candidate is tokenized x1 || suffix)
    population = []
    for i in range(N):
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=x1["input_ids"],  
                attention_mask=x1["attention_mask"],  
                max_length=x1_len+L+5,
                num_return_sequences=1,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id  # Set pad token ID to eos token ID
            )
        population.append(generated_tokens[:, 0:x1_len+L])

    for _ in range(G):
        # Mutation
        mutated_population = []
        for x in population:
            mutated_sample = greedy_coordinate_gradient(x, L, I, T, model, layer_num, sae, z_target, tokenizer, k, B)
            mutated_population.append(mutated_sample) # Best sequence after GCG, of shape (1, x1_len+suffix_len)
            
        # Selection
        ratios = []
        for x in mutated_population:
            z1_idx = sae.encode(model(x, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]).top_indices
            overlap = count_common(z1_idx, z2_idx) / len(z2_idx)
            ratios.append(overlap)

        top_half_idx = np.argsort(ratios)[-N//2:]
        selected_population = [mutated_population[idx] for idx in top_half_idx]

        # Recombination
        offspring = []
        for _ in range(N - len(selected_population)):
            parent1, parent2 = random.sample(selected_population, 2)
            # Select random break point in suffix
            crossover_point = random.randint(1, L)
            offspring_sample = parent1.clone()
            # Concatenate at the break point p1||p2
            offspring_sample[:, -crossover_point:] = parent2[:, -crossover_point:]
            offspring.append(offspring_sample)
        population = selected_population + offspring

    # Return the best result in the final population
    best_idx = -1
    best_ratio = 0
    for idx in range(N):
        z1_idx = sae.encode(model(population[idx], output_hidden_states=True).hidden_states[layer_num + 1][0][-1]).top_indices
        overlap = count_common(z1_idx, z2_idx) / len(z2_idx)
        if overlap > best_ratio:
            best_idx = idx
            best_ratio = overlap

    return tokenizer.decode(population[best_idx][0], skip_special_tokens=True), best_ratio

layer_num = 20
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
data_file = "./sae_samples_50.csv"
df = pd.read_csv(data_file)
sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)

# Index of evaluated sample in the data file
idx = 30
src_text = df.iloc[idx]['x1']
target_text = df.iloc[idx]['x2']

best_x1, best_ratio = evolutionary_algorithm(
    x1=tokenizer(src_text, return_tensors="pt").to(DEVICE),
    x2=tokenizer(target_text, return_tensors="pt").to(DEVICE),
    L=5, # suffix length
    model=model,
    layer_num=20, # target LLM layer
    sae=sae,
    tokenizer=tokenizer
)

print(f"Best adversarial input: {best_x1} with neuron overlap ratio {best_ratio}")