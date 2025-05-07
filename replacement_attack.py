import sys
sys.path.append('../sae')
from sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def count_common(x, y):
    return sum(1 for elem in x if elem in y)

# def run_individual_targeted_replace_attack(args):
#     run_individual_replace_attack(args, activate=args.individual_activate, targeted=True)

# def run_individual_untargeted_replace_attack(args):
#     run_individual_replace_attack(args, activate=args.individual_activate, targeted=False)

def run_individual_replace_attack(args):
    df = pd.read_csv(args.data_file)
    sae = Sae.load_from_disk(args.base_dir + f"layers.{args.layer_num}").to(DEVICE)
    log_file_path = f"./results/llama3-8b-generated/layer-{args.layer_num}/individual-replace-{args.sample_idx}-{'activate' if activate else 'deactivate'}-{'targeted' if targeted else 'untargeted'}.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = open(log_file_path, "w")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
    model = LlamaForCausalLM.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    x1_raw_text = df.iloc[args.sample_idx]['x1'][:-1]
    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x1_raw_processed = tokenizer(x1_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

    h1_raw = model(x1_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
    z1_signed = sae.encoder(h1_raw)
    s1 = sae.encode(h1_raw).top_indices
    s1_acts = sae.encode(h1_raw).top_acts

    if targeted:
        x2_raw_text = df.iloc[args.sample_idx]['x2'][:-1]
        x2_raw_processed = tokenizer(x2_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
        h2 = model(x2_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
        z2_signed = sae.encoder(h2)
        s2 = sae.encode(h2).top_indices

    neuron_list = (
        torch.topk(-z1_signed, k=args.num_selected).indices if activate
        else s1[torch.topk(s1_acts, k=args.num_selected).indices]
    ) if not targeted else (
        torch.topk(-z2_signed, k=args.num_selected).indices if activate
        else s2[torch.topk(sae.encode(h2).top_acts, k=args.num_selected).indices]
    )

    for t in range(1, x1_raw.shape[-1]):
        success_count = 0
        count = 0
        all_final_ranks = []
        for n_id in neuron_list:
            x1 = x1_raw_processed.clone()
            best_rank = z1_signed.shape[-1] if activate else 0
            for i in range(args.num_iters):
                with torch.no_grad():
                    out = model(x1, output_hidden_states=True)
                embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
                h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
                z1 = sae.pre_acts(h1)
                loss = torch.nn.functional.log_softmax(z1, dim=0)[n_id] if activate else -torch.nn.functional.log_softmax(z1, dim=0)[n_id]
                gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
                dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
                dot_prod[:, tokenizer.eos_token_id] = -float('inf')
                top_k_adv = (torch.topk(dot_prod, args.k).indices)[t]

                x1_batch = x1.repeat(args.k, 1)
                x1_batch[:, t] = top_k_adv

                with torch.no_grad():
                    new_embeds = model(x1_batch, output_hidden_states=True).hidden_states[0]
                    h1_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[args.layer_num + 1]
                    z1_batch = sae.pre_acts(h1_batch[:, -1, :])

                sorted_indices = torch.sort(z1_batch, dim=1, descending=True).indices
                rank_batch = (sorted_indices == n_id).nonzero(as_tuple=False)
                rank_per_sample = torch.full((z1_batch.shape[0],), z1_batch.shape[1], dtype=torch.long, device=z1_batch.device)
                rank_per_sample[rank_batch[:, 0]] = rank_batch[:, 1]
                best_idx = torch.argmin(rank_per_sample) if activate else torch.argmax(rank_per_sample)
                x1 = x1_batch[best_idx].unsqueeze(0)
                current_rank = rank_per_sample[best_idx].item()
                if (activate and current_rank < best_rank) or (not activate and current_rank > best_rank):
                    best_rank = current_rank
                if (activate and best_rank < len(s1)) or (not activate and best_rank > len(s1)):
                    success_count += 1
                    break
            all_final_ranks.append(best_rank)
            count += 1
        print(f"Token {t}: {success_count} out of {count} successful")
        print(f"Token {t} Final ranks: {all_final_ranks}")
    sys.stdout.close()

def run_group_targeted_replace_attack(args):
    df = pd.read_csv(args.data_file)
    sae = Sae.load_from_disk(args.base_dir + f"layers.{args.layer_num}").to(DEVICE)
    log_file_path = f"./results/llama3-8b-generated/layer-{args.layer_num}/group-targeted-replace-{args.sample_idx}.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = open(log_file_path, "w")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
    model = LlamaForCausalLM.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    x1_raw_text = df.iloc[args.sample_idx]['x1'][:-1]
    x2_raw_text = df.iloc[args.sample_idx]['x2'][:-1]
    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x2_raw = tokenizer(x2_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)

    x1_raw_processed = tokenizer(x1_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
    x2_raw_processed = tokenizer(x2_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

    h1 = model(x1_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
    h2 = model(x2_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
    z1 = sae.pre_acts(h1)
    z2 = sae.pre_acts(h2)

    best_loss = float("inf")
    x1_best = x1_raw.clone()
    for t in range(1, x1_raw.shape[-1]):
        for i in range(args.num_iters):
            x1_batch = x1_raw.repeat(args.k, 1)
            embeds = model.get_input_embeddings()(x1_batch).detach().requires_grad_(True)
            h1_batch = model(inputs_embeds=embeds, output_hidden_states=True).hidden_states[args.layer_num + 1][:, -1, :]
            z1_batch = sae.pre_acts(h1_batch)
            loss = -F.cosine_similarity(z1_batch, z2.unsqueeze(0)).mean()
            gradients = torch.autograd.grad(outputs=loss, inputs=embeds)[0]
            dot_prod = torch.matmul(gradients[:, t, :], model.get_input_embeddings().weight.T)
            top_k_adv = torch.topk(dot_prod, args.k, dim=-1).indices
            sampled = top_k_adv[torch.arange(args.k), torch.randint(0, args.k, (args.k,))]
            x1_batch[:, t] = sampled

            with torch.no_grad():
                h1_new = model(inputs_embeds=model.get_input_embeddings()(x1_batch), output_hidden_states=True).hidden_states[args.layer_num + 1][:, -1, :]
                z1_new = sae.pre_acts(h1_new)
                new_loss = -F.cosine_similarity(z1_new, z2.unsqueeze(0)).mean()
                print(f"Token {t} Iteration {i+1} Cosine Loss: {new_loss.item():.4f}")
                if new_loss < best_loss:
                    best_loss = new_loss
                    x1_best = x1_batch[0]
    print("Best adversarial x1:")
    print(tokenizer.decode(x1_best, skip_special_tokens=True))
    sys.stdout.close()

def run_group_untargeted_replace_attack(args):
    df = pd.read_csv(args.data_file)
    sae = Sae.load_from_disk(args.base_dir + f"layers.{args.layer_num}").to(DEVICE)
    log_file_path = f"./results/llama3-8b-generated/layer-{args.layer_num}/group-untargeted-replace-{args.sample_idx}.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = open(log_file_path, "w")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
    model = LlamaForCausalLM.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    x1_raw_text = df.iloc[args.sample_idx]['x1'][:-1]
    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x1_raw_processed = tokenizer(x1_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

    h1 = model(x1_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
    z1 = sae.pre_acts(h1)
    best_loss = -1
    x1_best = x1_raw.clone()
    for t in range(1, x1_raw.shape[-1]):
        for i in range(args.num_iters):
            x1_batch = x1_raw.repeat(args.k, 1)
            embeds = model.get_input_embeddings()(x1_batch).detach().requires_grad_(True)
            h1_batch = model(inputs_embeds=embeds, output_hidden_states=True).hidden_states[args.layer_num + 1][:, -1, :]
            z1_batch = sae.pre_acts(h1_batch)
            loss = -F.cosine_similarity(z1_batch, z1.unsqueeze(0)).mean()
            gradients = torch.autograd.grad(outputs=loss, inputs=embeds)[0]
            dot_prod = torch.matmul(gradients[:, t, :], model.get_input_embeddings().weight.T)
            top_k_adv = torch.topk(dot_prod, args.k, dim=-1).indices
            sampled = top_k_adv[torch.arange(args.k), torch.randint(0, args.k, (args.k,))]
            x1_batch[:, t] = sampled

            with torch.no_grad():
                h1_new = model(inputs_embeds=model.get_input_embeddings()(x1_batch), output_hidden_states=True).hidden_states[args.layer_num + 1][:, -1, :]
                z1_new = sae.pre_acts(h1_new)
                new_loss = -F.cosine_similarity(z1_new, z1.unsqueeze(0)).mean()
                print(f"Token {t} Iteration {i+1} Cosine Drop: {new_loss.item():.4f}")
                if new_loss > best_loss:
                    best_loss = new_loss
                    x1_best = x1_batch[0]
    print("Most different adversarial x1:")
    print(tokenizer.decode(x1_best, skip_special_tokens=True))
    sys.stdout.close()

