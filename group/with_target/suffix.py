import sys
sys.path.append('../sae')
from sae import Sae
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import argparse
from torch import nn, Tensor
from huggingface_hub import hf_hub_download, login
import transformers

print(transformers.__version__)

login(token="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")

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

def jump_relu(x, theta):
    return x * (x > theta).float()

def count_common(x, y):
    num_common = 0
    for elem in x:
        if elem in y:
            num_common += 1
    return num_common

def launch_attack(args):
    if args.dataset_type == "generated":
        df = pd.read_csv("./two_class_generated.csv")
    elif args.dataset_type == "news":
        df = pd.read_csv("./ag_news.csv")
    sample_idx = args.sample_idx
    layer_num = args.layer_num
    
    if args.model_type == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
        model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
        model.config.pad_token_id = model.config.eos_token_id
        sae = Sae.load_from_disk(BASE_DIR + f"layers.{layer_num}").to(DEVICE)
    elif args.model_type == "gemma2-2b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
        # sae, cfg_dict, sparsity = SAE.from_pretrained(
        #     release="gemma-2b-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        #     sae_id="blocks.12.hook_resid_post",  
        #     device=DEVICE
        # )
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_12/width_131k/average_l0_129/params.npz",
            force_download=False,
            cache_dir = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/gemma2-2b-sae"
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        W_enc = pt_params['W_enc']
        b_enc = pt_params['b_enc']
        theta = pt_params['threshold']
        num_top = args.num_top

    elif args.model_type == "gemma2-9b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-9b-pt-res",
            filename="layer_30/width_131k/average_l0_170/params.npz",
            force_download=False,
            cache_dir = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/gemma2-9b-sae"
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        W_enc = pt_params['W_enc']
        b_enc = pt_params['b_enc']
        theta = pt_params['threshold']
        num_top = args.num_top
    x1_raw_text = df.iloc[sample_idx]['x1'][:-1]
    x2_raw_text = df.iloc[sample_idx]['x2'][:-1]
    print(f"x1: {x1_raw_text}")
    print(f"x2: {x2_raw_text}")

    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x2_raw = tokenizer(x2_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x1_raw_processed = tokenizer(x1_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
    x2_raw_processed = tokenizer(x2_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

    # model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
    # model.config.pad_token_id = model.config.eos_token_id

    h1_raw = model(x1_raw_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
    if args.model_type == "llama3":
        z1_raw = sae.pre_acts(h1_raw)
        s1_raw = sae.encode(h1_raw).top_indices
        s1_raw_acts = sae.encode(h1_raw).top_acts
    elif args.model_type == "gemma2-2b" or args.model_type == "gemma2-9b":
        z1_raw = jump_relu(h1_raw @ W_enc + b_enc, theta)
        sorted, indices = torch.sort(z1_raw, descending=True)
        s1_raw = indices[:num_top]
        s1_raw_acts = sorted[:num_top]
        print(f"z1_raw: {indices[:50]}")
        print(f"z1_raw: {sorted[:50]}")
        # s1_raw = torch.nonzero(z1_raw > 0, as_tuple=True)[0]
        # s1_raw_acts = z1_raw[s1_raw]

    h2 = model(x2_raw_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
    if args.model_type == "llama3":
        z2 = sae.pre_acts(h2)
        s2 = sae.encode(h2).top_indices
        s2_acts = sae.encode(h2).top_acts
    elif args.model_type == "gemma2-2b" or args.model_type == "gemma2-9b":
        z2 = jump_relu(h2 @ W_enc + b_enc, theta)
        sorted, indices = torch.sort(z2, descending=True)
        print(f"z2: {indices[:50]}")
        print(f"z2: {sorted[:50]}")
        s2 = indices[:num_top]
        s2_acts = sorted[:num_top]
        # s2 = torch.nonzero(z2 > 0, as_tuple=True)[0]
        # s2_acts = z2[s2]
    initial_overlap = count_common(s1_raw, s2) / len(s2)
    print(f"Initial overlap = {initial_overlap}")
    

    num_iters = args.num_iters
    k = args.k
    num_adv = args.num_adv
    batch_size = args.batch_size
    num_top = args.num_top

    model.to(DEVICE)
    model.eval()
    # best_loss = float("Inf")
    best_overlap = 0.0
    losses = []
    overlaps = []
    inputs = tokenizer(df.iloc[sample_idx]['x1'][:-1], return_tensors="pt")
    x1_init = model.generate(
        input_ids=inputs['input_ids'].to(DEVICE),
        attention_mask=inputs['attention_mask'].to(DEVICE),
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
    x1_init_processed = tokenizer(x1_init_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
    # print(f"x1 init processed: {x1_init_processed}", x1_init_processed.shape)
    h1 = model(x1_init_processed, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
    if args.model_type == "llama3":
        z1 = sae.pre_acts(h1)
        s1 = sae.encode(h1).top_indices
        s1_acts = sae.encode(h1).top_acts
    elif args.model_type == "gemma2-2b" or args.model_type == "gemma2-9b":
        # z1 = nn.functional.relu(h1 @ sae.W_enc + sae.b_enc)
        z1 = jump_relu(h1 @ W_enc + b_enc, theta)
        sorted, indices = torch.sort(z1, descending=True)
        s1 = indices[:num_top]
        s1_acts = sorted[:num_top]
        print(f"z1: {indices[:50]}")
        print(f"z1: {sorted[:50]}")
        
    print(f"s1 s2 overlap = {count_common(s1, s2) / len(s2)}")
    x1 = x1_init_processed
    best_x1 = x1_init[0]
    for i in range(num_iters):
        with torch.no_grad():
            # out = model(x1, output_hidden_states=True)
            embeddings = model.get_input_embeddings()(x1)
        
        # embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
        embeddings = embeddings.detach().clone().requires_grad_(True)
        
        # Forward pass again with embeddings
        h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
        if args.model_type == "llama3":
            z1 = sae.pre_acts(h1)
        elif args.model_type == "gemma2-2b" or args.model_type == "gemma2-9b":
            # z1 = nn.functional.relu(h1 @ sae.W_enc + sae.b_enc)
            z1 = jump_relu(h1 @ W_enc + b_enc, theta)
        
        # Compute similarity loss
        loss = cos_sim(z1, z2)

        gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=False)[0]
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
            if args.model_type == "llama3":
                z1_batch = sae.pre_acts(h1_batch[:, -1, :])
                s1_batch = sae.encode(h1_batch[:, -1, :]).top_indices # (k, num_topk=192)
            elif args.model_type == "gemma2-2b" or args.model_type == "gemma2-9b":
                # z1_batch = nn.functional.relu(h1_batch[:, -1, :] @ sae.W_enc + sae.b_enc)
                z1_batch = jump_relu(h1_batch[:, -1, :] @ W_enc + b_enc, theta)
                sorted, indices = torch.sort(z1_batch, dim=-1, descending=True)
                s1_batch = indices[:, :num_top]
                s1_acts_batch = sorted[:, :num_top]
                # s1_batch = []
                # for b in range(z1_batch.size(0)):
                #     s1 = torch.nonzero(z1_batch[b] > 0, as_tuple=True)[0]
                #     s1_batch.append(s1)
                # print(s1_batch)
        # s1_batch = sae.encode(h1_batch[:, -1, :]).top_indices # (k, num_topk=192)
        overlap_batch = torch.tensor([count_common(s1_batch[j], s2) / len(s2) for j in range(s1_batch.shape[0])])
        
        best_idx = torch.argmax(overlap_batch)
        print(f"s1_batch best: {s1_batch[best_idx][:50]}")
        print(f"s1_acts_batch best: {s1_acts_batch[best_idx][:50]}")
        x1 = x1_batch[best_idx].unsqueeze(0)
        # print(f"best idx = {best_idx}")
        if overlap_batch[best_idx] > best_overlap:
            best_overlap = overlap_batch[best_idx]
            best_x1 = x1[0][:x1_init.shape[-1]]
        best_loss = cos_sim(z1_batch[best_idx], z2).item()
        x1_text = tokenizer.decode(best_x1, skip_special_tokens=True)
        losses.append(best_loss)
        if type(best_overlap) == float:
            overlaps.append(best_overlap)
        else:
            overlaps.append(best_overlap.item())

        print(f"Iteration {i+1} loss = {best_loss}")
        print(f"Iteration {i+1} best overlap ratio = {best_overlap}")
        print(f"Iteration {i+1} input text: {x1_text}")
        print("--------------------")
        torch.cuda.empty_cache()

    # print(losses)
    # print(overlaps)
    # print(f"Best loss = {best_loss}")
    # print(f"Best overlap = {best_overlap}")
    # print(f"Best x1 = {tokenizer.decode(best_x1, skip_special_tokens=True)}")
    if args.model_type == "llama3":
        output_file = f"./results/llama3-8b-{args.dataset_type}/layer-20/targeted-population-suffix-{args.sample_idx}.txt"
    elif args.model_type == "gemma2-2b":
        output_file = f"./results/gemma2-2b-{args.dataset_type}/layer-12/targeted-population-suffix-{args.sample_idx}.txt"
    elif args.model_type == "gemma2-9b":
        output_file = f"./results/gemma2-9b-{args.dataset_type}/layer-30/targeted-population-suffix-{args.sample_idx}.txt"
    with open(output_file, "w") as f:
        f.write(f"Config:\n{args}\n\n")
        f.write(f"Initial overlap:\n{initial_overlap}\n\n")
        f.write(f"Losses across iterations:\n{losses}\n\n")
        f.write(f"Overlaps across iterations:\n{overlaps}\n\n")
        f.write(f"Best loss: {best_loss}\n")
        f.write(f"Best overlap: {best_overlap}\n")
        f.write(f"Best x1:\n{tokenizer.decode(best_x1, skip_special_tokens=True)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack Setting: Population/Targeted/Suffix")
    parser.add_argument("--model_type", type=str, default="llama3")
    parser.add_argument("--dataset_type", type=str, default="generated")
    parser.add_argument("--sample_idx", type=int, default=20)
    parser.add_argument("--layer_num", type=int, default=20)
    parser.add_argument("--num_adv", type=int, default=3)
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--num_top", type=int, default=20)
    parser.add_argument("--num_iters", type=int, default=50)

    args = parser.parse_args()
    launch_attack(args)