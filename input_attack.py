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

layer_num = 20
# sae = Sae.load_many_from_hub("EleutherAI/sae-llama-3-8b-32x")
sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)
# sae.device = DEVICE
# print(sae)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

input_src = tokenizer("The cat slept peacefully on the sunny windowsill", return_tensors="pt").to(DEVICE)
input_target = tokenizer("An astronaut floated weightlessly in the vast expanse of space", return_tensors="pt").to(DEVICE)

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
out_tokens_src = model.generate(
    input_src['input_ids'],
    max_length=20,  # Maximum length of the generated text
    temperature=0.0,
    do_sample=False,
    num_return_sequences=1,  # Number of sequences to generate
    no_repeat_ngram_size=2,  # To avoid repeating the same n-grams
    # early_stopping=True,  # Stop generating when it seems complete
)

# Decode the output
generation_src = tokenizer.decode(out_tokens_src[0], skip_special_tokens=True)
print(generation_src)
output_src = model(**input_src, output_hidden_states=True)
output_target = model(**input_target, output_hidden_states=True)
z = output_src.hidden_states[layer_num+1][0][-1]
# latent_acts = sae.encode(outputs.hidden_states[21][0][-1])
# print(latent_acts.top_acts.shape)
# print(latent_acts.top_acts)

latent_acts_src = sae.pre_acts(output_src.hidden_states[layer_num+1][0][-1])
latent_acts_target = sae.pre_acts(output_target.hidden_states[layer_num+1][0][-1])
# corr = torch.corrcoef(torch.cat([latent_acts_src, latent_acts_target], dim=0))

# latent_acts_src = sae.encode(output_src.hidden_states[21][0][-1])
top_idx_target = sae.encode(output_target.hidden_states[layer_num+1][0][-1]).top_indices
top_acts_target = sae.encode(output_target.hidden_states[layer_num+1][0][-1]).top_acts
target_mask = (latent_acts_target >= torch.amin(top_acts_target)).float()
# print(latent_acts_src.top_acts)
# print(latent_acts_target.top_acts)

num_iters = 20
k = 2000
num_adv = 1
batch_size = 2000
alpha = 0.1
attack_mode = 'suffix'
loss_func = torch.nn.MSELoss()

if attack_mode == 'suffix':
    input_src = torch.tensor(tokenizer.encode("The cat slept peacefully on the sunny windowsill")).unsqueeze(0).to(DEVICE)
elif attack_mode == 'prefix':
    input_src = torch.tensor(tokenizer.encode("The cat slept peacefully on the sunny windowsill")).unsqueeze(0).to(DEVICE)
model.to(DEVICE)
best_loss = 100.0
losses = []
overlaps = []
sign_agreements_ratio = []
print(f"Original Input: {tokenizer.decode(input_src[0], skip_special_tokens=True)}")

for i in range(num_iters):
    with torch.no_grad():
        out = model(input_src, output_hidden_states=True)
    embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
    lm_out = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[layer_num+1][0][-1]
    sae_out = sae.pre_acts(lm_out) 
    # loss = - cos_sim(sae_out, latent_acts_target) + alpha * torch.norm(lm_out - z)
    top_acts = sae.encode(lm_out).top_acts
    threshold = torch.amin(top_acts).detach()
    loss = loss_func(torch.sigmoid(sae_out - threshold), target_mask)
    gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
    dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
    # print(dot_prod.shape)

    cls_token_idx = tokenizer.encode('[CLS]')[1]
    sep_token_idx = tokenizer.encode('[SEP]')[1]
    dot_prod[:, cls_token_idx] = -float('inf')
    dot_prod[:, sep_token_idx] = -float('inf')

    # Get top k adversarial tokens
    if attack_mode == "suffix":
        top_k_adv = (torch.topk(dot_prod, k).indices)  
    elif attack_mode == "prefix":
        top_k_adv = (torch.topk(dot_prod, k).indices)
    # shape: (seq_len, k)

    tokens_batch = []
    for _ in range(batch_size):
        random_idx = torch.randint(0, top_k_adv.shape[0], (1,))
        random_top_k_idx = torch.randint(0, k, (1,))
        batch_item = input_src.clone().detach() # (1, seq_len)
        if attack_mode == "suffix":
            batch_item[0, random_idx] = top_k_adv[random_idx, random_top_k_idx]
        elif attack_mode == "prefix":
            # requires further debugging
            batch_item[0, random_idx] = top_k_adv[random_idx, random_top_k_idx]
        tokens_batch.append(batch_item)

    tokens_batch = torch.cat(tokens_batch, dim=0)

    with torch.no_grad():
        new_embeds = model(tokens_batch, output_hidden_states=True).hidden_states[0]
        model_out = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[layer_num+1]
        sae_out = sae.pre_acts(model_out[:, -1, :]) # (batch size, sae pre-act size)
        top_acts = sae.encode(model_out[:, -1, :]).top_acts # (batch size, sae top-act size)
        # masks = target_mask.repeat(sae_out.shape[0], 1)
        acts = torch.sigmoid(sae_out - torch.amin(top_acts, dim=-1).unsqueeze(-1))
        new_loss = torch.tensor([loss_func(acts[j], target_mask) for j in range(sae_out.shape[0])])
        # print(new_loss)
        # new_loss = torch.nn.functional.mse_loss(torch.sigmoid(sae_out - torch.amin(top_acts, dim=-1).unsqueeze(-1)), masks, reduction='none')
    
    best_idx = torch.argmin(new_loss)
    if new_loss[best_idx] < best_loss:
        best_loss = new_loss[best_idx]
        input_src = tokens_batch[best_idx].unsqueeze(0)
    # corr = cos_sim(out[max_idx], latent_acts_target)
    losses.append(best_loss.item())
    with torch.no_grad():
        out = model(input_src, output_hidden_states=True)
    top_idx_src = sae.encode(out.hidden_states[layer_num+1][0][-1]).top_indices
    overlap_ratio = count_common(top_idx_src, top_idx_target) / len(top_idx_src)
    overlaps.append(overlap_ratio)
    # new_acts = sae.pre_acts(out.hidden_states[21][0][-1])
    top_acts_src = sae.encode(out.hidden_states[layer_num+1][0][-1]).top_acts
    agree_ratio = torch.sum(torch.sign(top_acts_src == top_acts_target)) / len(top_acts_src)
    sign_agreements_ratio.append(agree_ratio.item())

    out_tokens = model.generate(
        input_src,
        max_length=20,  # Maximum length of the generated text
        temperature=0.0,
        do_sample=False,
        num_return_sequences=1,  # Number of sequences to generate
        no_repeat_ngram_size=2,  # To avoid repeating the same n-grams
        # early_stopping=True,  # Stop generating when it seems complete
    )
    generation = tokenizer.decode(out_tokens[0], skip_special_tokens=True)

    print(f"Iteration {i+1} loss = {best_loss.item()}")    
    print(f"Iteration {i+1} overlap_raio = {overlap_ratio}")   
    # print(f"Iteration {i+1} num_sign_agreements = {agree_ratio} out of {len(top_acts_src)}")  
    print(f"Iteration {i+1} input: {tokenizer.decode(input_src[0], skip_special_tokens=True)}")
    print(f"Iteration {i+1} text generation: {generation}")
    print("--------------------")
print(losses)
print(overlaps)
# print(sign_agreements_ratio)

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].plot(np.arange(1, num_iters+1), np.array(losses))
# axs[0].set_xlabel('Iteration')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Loss vs. Iteration')

# axs[1].plot(np.arange(1, num_iters+1), np.array(overlaps))
# axs[1].set_xlabel('Iteration')
# axs[1].set_ylabel('Neuron Overlap')
# axs[1].set_title('Neuron Overlap vs. Iteration')

# # axs[2].plot(np.arange(1, num_iters+1), sign_agreements_ratio)
# # axs[2].set_xlabel('Iteration')
# # axs[2].set_ylabel('Sign Agreements')
# # axs[2].set_title('Sign Agreements vs. Iteration')
# plt.savefig(f"./results/llama3-8b/layer-{layer_num}/sigmoid-1.png")
# plt.show()

        

