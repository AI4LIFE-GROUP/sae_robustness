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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
src_text = "The cat slept peacefully on the sunny windowsill"
target_text = "An astronaut floated weightlessly in the vast expanse of space"
input_src = tokenizer(src_text, return_tensors="pt").to(DEVICE)
input_target = tokenizer(target_text, return_tensors="pt").to(DEVICE)

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


# Decode the output src
generated_tokens_src = out_tokens_src[:, input_src['input_ids'].shape[-1]:]
# print(out_tokens_src.shape)
# print(generated_tokens_src.shape)
generated_text_src = tokenizer.batch_decode(generated_tokens_src, skip_special_tokens=True)[0]
print(generated_text_src)

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

num_iters = 50
k = 1000
num_adv = 10
batch_size = 800
alpha = 0.01
temp_max = 10
temp_min = 1
mode = 'suffix'
customized_loss = True

loss_func = torch.nn.MSELoss()
# input_src = torch.tensor(tokenizer.encode("The cat slept peacefully on the sunny windowsill")).unsqueeze(0).to(DEVICE)

model.to(DEVICE)
best_loss = float("Inf")
losses = []
overlaps = []
# sign_agreements_ratio = []
print(f"Original Input: {tokenizer.decode(input_src['input_ids'][0], skip_special_tokens=True)}")

prompt_text_src = src_text
if mode == 'suffix':
    prompt_text_src = "The cat slept peacefully on the sunny windowsill" + " *" * num_adv
print(prompt_text_src)
for i in range(num_iters):
    temp = temp_max - (temp_max - temp_min) / num_iters * i
    # print(temp)
    combined_text_src = prompt_text_src + generated_text_src
    combined_tokens_src = tokenizer(combined_text_src, return_tensors="pt").to(DEVICE)['input_ids']
    prompt_tokens_src = tokenizer(prompt_text_src, return_tensors="pt").to(DEVICE)['input_ids']
    prompt_src_token_len = prompt_tokens_src.shape[-1]
    # generated_tokens_src = tokenizer(generated_text_src, return_tensors="pt").to(DEVICE)['input_ids']
    generated_tokens_src = combined_tokens_src[:, prompt_src_token_len:]
    # print(combined_tokens_src, prompt_tokens_src, generated_tokens_src)
    # print(combined_tokens_src.shape, prompt_tokens_src.shape, generated_tokens_src.shape)
    with torch.no_grad():
        out = model(combined_tokens_src, output_hidden_states=True)
    embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
    lm_out = model(inputs_embeds=embeddings, output_hidden_states=True)
    lm_h = lm_out.hidden_states[layer_num+1][0][prompt_src_token_len-1] # (,hidden_dim)
    lm_logits = lm_out.logits # (batch_size, seq_len, vocab_size)
    # print(lm_h.shape, lm_logits.shape, out_tokens_src.shape)
    # lm_match_loss = F.cross_entropy(lm_logits[0, prompt_src_token_len:, :], generated_tokens_src[0])
    # print(lm_match_loss)
    sae_out = sae.pre_acts(lm_h) # (, sae_dim)
    top_acts = sae.encode(lm_h).top_acts
    threshold = torch.amin(top_acts).detach()
    # print(sae_out.shape, threshold.shape, target_mask.shape)
    target_mask[torch.argwhere(target_mask == 0)] = -1
    # print(torch.argwhere(target_mask == 1))
    # print(target_mask[:100])
    # sae_match_loss = loss_func(torch.sigmoid((sae_out - threshold) / temp), target_mask)
    sae_match_loss = - torch.sum(sae_out * target_mask)
    # print(sae_match_loss)
    # loss = sae_match_loss + alpha * lm_match_loss
    loss = sae_match_loss
    # print(loss)

    gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
    dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
    # print(dot_prod.shape)

    del embeddings, lm_out, lm_h, lm_logits, sae_out, top_acts

    cls_token_idx = tokenizer.encode('[CLS]')[1]
    sep_token_idx = tokenizer.encode('[SEP]')[1]
    dot_prod[:, cls_token_idx] = -float('inf')
    dot_prod[:, sep_token_idx] = -float('inf')

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, k).indices)[:prompt_src_token_len]
    # print(top_k_adv.shape)  
    # shape: (prompt_len, k)

    prompt_tokens_batch = []
    for _ in range(batch_size):
        if mode == 'suffix':
            random_idx = torch.randint(0, num_adv, (1,))
            random_top_k_idx = torch.randint(0, k, (1,))
            batch_item = prompt_tokens_src.clone().detach() # (1, seq_len)
            batch_item[0, -num_adv:][random_idx] = top_k_adv[random_idx, random_top_k_idx]
        prompt_tokens_batch.append(batch_item)
        

    prompt_tokens_batch = torch.cat(prompt_tokens_batch, dim=0) # (batch_size, prompt_len)

    with torch.no_grad():
        continuation_tokens_batch = generated_tokens_src.repeat(batch_size, 1)
        tokens_batch = torch.cat([prompt_tokens_batch, continuation_tokens_batch], dim=-1) # (batch_size, seq_len)
        new_combined_tokens_len = tokens_batch.shape[-1]
        # new_embeds = model(tokens_batch, output_hidden_states=True).hidden_states[0]
        new_lm_out = model(tokens_batch, output_hidden_states=True)
        new_lm_h = new_lm_out.hidden_states[layer_num+1][:, prompt_src_token_len-1, :] # (batch_size, hidden_dim)
        new_lm_logits = new_lm_out.logits[:, prompt_src_token_len:, :] # (batch_size, continuation_len, vocab_size)
        new_sae_out = sae.pre_acts(new_lm_h) # (batch_size, sae_dim)
        new_top_acts = sae.encode(new_lm_h).top_acts # (batch_size, sae_k)
        new_top_idx = sae.encode(new_lm_h).top_indices # (batch_size, sae_k)
        new_losses = []

        # print(new_lm_logits.shape, continuation_tokens_batch.shape)
        for j in range(batch_size):
            # new_lm_match_loss = F.cross_entropy(new_lm_logits[j, :, :], continuation_tokens_batch[j])
            threshold = torch.amin(new_top_acts[j]).detach()
            # new_sae_match_loss = loss_func(torch.sigmoid(new_sae_out[j] - threshold), target_mask)
            new_sae_match_loss = - torch.sum(new_sae_out * target_mask)
            # new_loss = new_sae_match_loss + alpha * new_lm_match_loss
            new_loss = new_sae_match_loss
            new_losses.append(new_loss)
    
    best_idx = torch.argmin(torch.tensor(new_losses))
    best_prompt_tokens_src = prompt_tokens_src
    # if new_losses[best_idx] < best_loss:
    best_loss = new_losses[best_idx]
    best_prompt_tokens_src = prompt_tokens_batch[best_idx].unsqueeze(0) # (1, prompt_len)

    del tokens_batch, new_lm_out, new_lm_h, new_lm_logits, new_sae_out, new_top_acts, continuation_tokens_batch
    
    if customized_loss:
        losses.append(best_loss.item())
    else:
        losses.append(best_loss.item())
    overlap_ratio = count_common(new_top_idx[best_idx], top_idx_target) / len(top_idx_target)
    overlaps.append(overlap_ratio)
#     # new_acts = sae.pre_acts(out.hidden_states[21][0][-1])
#     top_acts_src = sae.encode(out.hidden_states[layer_num+1][0][-1]).top_acts
#     agree_ratio = torch.sum(torch.sign(top_acts_src == top_acts_target)) / len(top_acts_src)
#     sign_agreements_ratio.append(agree_ratio.item())

    new_out_tokens = model.generate(
        best_prompt_tokens_src,
        max_length=new_combined_tokens_len,  # Maximum length of the generated text
        temperature=0.0,
        do_sample=False,
        num_return_sequences=1,  # Number of sequences to generate
        no_repeat_ngram_size=2,  # To avoid repeating the same n-grams
        # early_stopping=True,  # Stop generating when it seems complete
    )
    new_generation = tokenizer.batch_decode(new_out_tokens, skip_special_tokens=True)[0]
    prompt_text_src = tokenizer.batch_decode(best_prompt_tokens_src, skip_special_tokens=True)[0]
    if customized_loss:
        print(f"Iteration {i+1} loss = {best_loss}")
    else:
        print(f"Iteration {i+1} loss = {best_loss.item()}")       
    print(f"Iteration {i+1} overlap_raio = {overlap_ratio}")   
    # print(f"Iteration {i+1} num_sign_agreements = {agree_ratio} out of {len(top_acts_src)}")  
    print(f"Iteration {i+1} input: {prompt_text_src}")
    print(f"Iteration {i+1} text generation: {new_generation}")
    print("--------------------")
print(losses)
print(overlaps)
# print(sign_agreements_ratio)

# # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# # axs[0].plot(np.arange(1, num_iters+1), np.array(losses))
# # axs[0].set_xlabel('Iteration')
# # axs[0].set_ylabel('Loss')
# # axs[0].set_title('Loss vs. Iteration')

# # axs[1].plot(np.arange(1, num_iters+1), np.array(overlaps))
# # axs[1].set_xlabel('Iteration')
# # axs[1].set_ylabel('Neuron Overlap')
# # axs[1].set_title('Neuron Overlap vs. Iteration')

# # # axs[2].plot(np.arange(1, num_iters+1), sign_agreements_ratio)
# # # axs[2].set_xlabel('Iteration')
# # # axs[2].set_ylabel('Sign Agreements')
# # # axs[2].set_title('Sign Agreements vs. Iteration')
# # plt.savefig(f"./results/llama3-8b/layer-{layer_num}/sigmoid-1.png")
# # plt.show()

        

