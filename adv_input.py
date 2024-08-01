from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/sae/"
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

# sae = Sae.load_many_from_hub("EleutherAI/sae-llama-3-8b-32x")
sae = Sae.load_from_disk(BASE_DIR + "layers.20").to(DEVICE)
# sae.device = DEVICE
# print(sae)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

input_src = tokenizer("The cat slept peacefully on the sunny windowsill ", return_tensors="pt").to(DEVICE)
input_target = tokenizer("An astronaut floated weightlessly in the vast expanse of space ", return_tensors="pt").to(DEVICE)


model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)

output_src = model(**input_src, output_hidden_states=True)
output_target = model(**input_target, output_hidden_states=True)
# latent_acts = sae.encode(outputs.hidden_states[21][0][-1])
# print(latent_acts.top_acts.shape)
# print(latent_acts.top_acts)

latent_acts_src = sae.pre_acts(output_src.hidden_states[21][0][-1])
latent_acts_target = sae.pre_acts(output_target.hidden_states[21][0][-1])
# corr = torch.corrcoef(torch.cat([latent_acts_src, latent_acts_target], dim=0))

# latent_acts_src = sae.encode(output_src.hidden_states[21][0][-1])
top_idx_target = sae.encode(output_target.hidden_states[21][0][-1]).top_indices
# print(latent_acts_src.top_acts)
# print(latent_acts_target.top_acts)

num_iters = 30
k = 500
num_adv = 5
batch_size = 1000
attack_mode = 'prefix'

if attack_mode == 'suffix':
    input_src = torch.tensor(tokenizer.encode("The cat slept peacefully on the sunny windowsill " + ('* ' * num_adv))).unsqueeze(0).to(DEVICE)
elif attack_mode == 'prefix':
    input_src = torch.tensor(tokenizer.encode('*' + (' *' * (num_adv-1)) + " The cat slept peacefully on the sunny windowsill")).unsqueeze(0).to(DEVICE)
model.to(DEVICE)
best_loss = 100.0
similarities = []
overlaps = []
num_sign_agreements = []
print(f"Original Input: {tokenizer.decode(input_src[0], skip_special_tokens=True)}")

for i in range(num_iters):
    with torch.no_grad():
        out = model(input_src, output_hidden_states=True)
    embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
    lm_out = model(inputs_embeds=embeddings, output_hidden_states=True)
    sae_out = sae.pre_acts(lm_out.hidden_states[21][0][-1])
    loss = -cos_sim(sae_out, latent_acts_target)
    gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
    dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
    # print(dot_prod.shape)

    cls_token_idx = tokenizer.encode('[CLS]')[1]
    sep_token_idx = tokenizer.encode('[SEP]')[1]
    dot_prod[:, cls_token_idx] = -float('inf')
    dot_prod[:, sep_token_idx] = -float('inf')

    # Get top k adversarial tokens
    if attack_mode == "suffix":
        top_k_adv = (torch.topk(dot_prod, k).indices)[-num_adv-1:-1]    
    elif attack_mode == "prefix":
        top_k_adv = (torch.topk(dot_prod, k).indices)[1:num_adv+1]

    tokens_batch = []
    for _ in range(batch_size):
        random_idx = torch.randint(0, num_adv, (1,)).to(DEVICE)
        random_top_k_idx = torch.randint(0, k, (1,)).to(DEVICE)
        batch_item = input_src.clone().detach()
        if attack_mode == "suffix":
            batch_item[0, -num_adv-1:-1][random_idx] = top_k_adv[random_idx, random_top_k_idx]
        elif attack_mode == "prefix":
            # requires further debugging
            batch_item[0, 1:num_adv+1][random_idx] = top_k_adv[random_idx, random_top_k_idx]
        tokens_batch.append(batch_item)

    tokens_batch = torch.cat(tokens_batch, dim=0)

    with torch.no_grad():
        new_embeds = model(tokens_batch, output_hidden_states=True).hidden_states[0]
        out = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[21]
        out = sae.pre_acts(out[:, -1, :])
    new_loss = torch.tensor([-cos_sim(out[j], latent_acts_target) for j in range(out.shape[0])])
    best_idx = torch.argmin(new_loss)
    if new_loss[best_idx] < best_loss:
        best_loss = new_loss[best_idx]
        input_src = tokens_batch[best_idx].unsqueeze(0)
    # corr = cos_sim(out[max_idx], latent_acts_target)
    similarities.append(-best_loss.item())
    with torch.no_grad():
        out = model(input_src, output_hidden_states=True)
    top_idx_src = sae.encode(out.hidden_states[21][0][-1]).top_indices
    num_overlap = count_common(top_idx_src, top_idx_target)
    overlaps.append(num_overlap)
    new_acts = sae.pre_acts(out.hidden_states[21][0][-1])
    num_agrees = torch.sum(torch.sign(new_acts == latent_acts_target))
    num_sign_agreements.append(num_agrees.item())
    print(f"Iteration {i+1} similarity = {-best_loss.item()}")    
    print(f"Iteration {i+1} num_overlap = {num_overlap}")   
    print(f"Iteration {i+1} num_sign_agreements = {num_agrees} out of {new_acts.shape[0]}")  
    print(f"Iteration {i+1} input: {tokenizer.decode(input_src[0], skip_special_tokens=True)}")
    print("--------------------")
print(similarities)
print(overlaps)
print(num_sign_agreements)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(np.arange(1, num_iters+1), np.array(similarities))
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Similarity')
axs[0].set_title('Cosine Similarity (Raw Activations) vs. Iteration')

axs[1].plot(np.arange(1, num_iters+1), np.array(overlaps))
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Neuron Overlap')
axs[1].set_title('Neuron Overlap vs. Iteration')

axs[2].plot(np.arange(1, num_iters+1), num_sign_agreements)
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Sign Agreements')
axs[2].set_title('Sign Agreements vs. Iteration')
plt.savefig("./results/llama3-8b/layer-20/500_5_1000-prefix-2.png")
plt.show()

        

