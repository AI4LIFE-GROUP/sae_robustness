from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/sae/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

def cos_sim(tensor1, tensor2):
    # Normalize the tensors
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    
    # Compute the dot product
    dot_product = torch.sum(tensor1 * tensor2)
    
    # Compute the cosine similarity
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return cosine_similarity

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
# latent_acts_target = sae.encode(output_target.hidden_states[21][0][-1])
# print(latent_acts_src.top_acts)
# print(latent_acts_target.top_acts)
sim = cos_sim(latent_acts_src, latent_acts_target)
print(sim)

num_iters = 100
k = 200
num_adv = 5
batch_size = 500

input_src = torch.tensor(tokenizer.encode("The cat slept peacefully on the sunny windowsill " + ('* ' * num_adv))).unsqueeze(0).to(DEVICE)
model.to(DEVICE)
best_loss = 100.0
for i in range(num_iters):
    with torch.no_grad():
        out = model(input_src, output_hidden_states=True)
    embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
    lm_out = model(inputs_embeds=embeddings, output_hidden_states=True)
    sae_out = sae.pre_acts(lm_out.hidden_states[21][0][-1])
    loss = -cos_sim(sae_out, latent_acts_target)
    gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
    dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
    print(dot_prod.shape)

    cls_token_idx = tokenizer.encode('[CLS]')[1]
    sep_token_idx = tokenizer.encode('[SEP]')[1]
    dot_prod[:, cls_token_idx] = -float('inf')
    dot_prod[:, sep_token_idx] = -float('inf')

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, k).indices)[-num_adv-1:-1]    

    tokens_batch = []
    for _ in range(batch_size):
        random_idx = torch.randint(0, num_adv, (1,)).to(DEVICE)
        random_top_k_idx = torch.randint(0, k, (1,)).to(DEVICE)
        batch_item = input_src.clone().detach()
        batch_item[0, -num_adv-1:-1][random_idx] = top_k_adv[random_idx, random_top_k_idx]
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
    
    print(f"Iteration {i+1} similarity = {-best_loss.item()}")    
    print(f"Iteration {i+1} input: {tokenizer.decode(input_src[0])}")

        

