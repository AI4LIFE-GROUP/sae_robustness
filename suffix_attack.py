import sys
sys.path.append('../sae')
from sae import Sae
from utils import *
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

# def run_individual_targeted_suffix_attack(args):
#     run_individual_suffix_attack(args, activate=args.individual_activate, targeted=True)

# def run_individual_untargeted_suffix_attack(args):
#     run_individual_suffix_attack(args, activate=args.individual_activate, targeted=False)

def run_individual_suffix_attack(args):
    df = pd.read_csv(args.data_file)
    model, tokenizer, sae = load_model_and_sae(args.model_type, args.layer_num)
    if args.log:
        log_file_path = f"./results/{args.model_type}/layer-{args.layer_num}/{args.targeted}-individual-suffix-{args.sample_idx}-{'activate' if args.activate else 'deactivate'}.txt"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        sys.stdout = open(log_file_path, "w")

    x1_raw_text = df.iloc[args.sample_idx]['x1'][:-1]
    print(f"x1: {x1_raw_text}")
    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x1_raw_processed = tokenizer(x1_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

    h1_raw = model(x1_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
    if args.model_type == "llama3-8b":
        z1_signed = sae.encoder(h1_raw)
    elif args.model_type == "gemma2-9b":
        z1_signed = h1_raw @ sae['W_enc'] + sae['b_enc']
    z1_raw, s1_raw, s1_acts_raw = extract_sae_features(h1_raw, sae, args.model_type, args.k)

    x1_init = model.generate(x1_raw, max_length=x1_raw.shape[-1] + args.suffix_len, do_sample=False, num_return_sequences=1)
    x1_init_text = tokenizer.decode(x1_init[0], skip_special_tokens=True)
    x1_init_processed = tokenizer(x1_init_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

    h1 = model(x1_init_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
    z1, s1, s1_acts = extract_sae_features(h1, sae, args.model_type, args.k)

    if args.targeted:
        x2_raw_text = df.iloc[args.sample_idx]['x2'][:-1]
        print(f"x2: {x2_raw_text}")
        x2_raw_processed = tokenizer(x2_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
        h2 = model(x2_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
        z2, s2, s2_acts = extract_sae_features(h2, sae, args.model_type, args.k)
        print(f"Initial s1_raw s2 overlap = {count_common(s1_raw, s2) / len(s2)}")
        print(f"Initial s1 s2 overlap = {count_common(s1, s2) / len(s2)}")
        if args.activate:
            # Neurons in s2 but not in s1
            mask = ~torch.isin(s2, s1)
            top_indices = torch.topk(s2_acts[mask], k=min(args.num_latents, s2_acts[mask].numel())).indices
            neuron_list = s2[mask][top_indices]
        else:
            # Neurons in s1 but not in s2
            mask = ~torch.isin(s1, s2)
            top_indices = torch.topk(s1_acts[mask], k=min(args.num_latents, s1_acts[mask].numel())).indices
            neuron_list = s1[mask][top_indices]
    else:
        print(f"Initial s1_raw s1 overlap = {count_common(s1_raw, s1) / len(s1)}")
        neuron_list = torch.topk(-z1_signed, k=args.num_latents).indices if args.activate else s1[torch.topk(s1_acts, k=args.num_latents).indices]
    print(len(neuron_list))
    success_count, count = 0, 0
    all_final_ranks = []
    for n_id in neuron_list:
        x1 = x1_init_processed.clone()
        best_rank = z1.shape[-1] if args.activate else 0

        for i in range(args.num_iters):
            with torch.no_grad():
                out = model(x1, output_hidden_states=True)
            embeddings = out.hidden_states[0].clone().detach().requires_grad_(True)
            h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
            z1, s1, s1_acts = extract_sae_features(h1, sae, args.model_type, args.k)
            loss = torch.nn.functional.log_softmax(z1, dim=0)[n_id] if args.activate else -torch.nn.functional.log_softmax(z1, dim=0)[n_id]
            gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=True)[0]
            dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
            dot_prod[:, tokenizer.eos_token_id] = -float('inf')
            top_m_adv = (torch.topk(dot_prod, args.m).indices)[x1_init.shape[-1] - args.suffix_len:x1_init.shape[-1]]

            x1_batch = x1.repeat(args.batch_size, 1).clone()
            rand_token_idx = torch.randint(0, args.suffix_len, (args.batch_size,))
            rand_top_m_idx = torch.randint(0, args.m, (args.batch_size,))
            batch_indices = torch.arange(args.batch_size)
            x1_batch[:, x1_init.shape[-1] - args.suffix_len:x1_init.shape[-1]][batch_indices, rand_token_idx] = top_m_adv[0, rand_top_m_idx]

            with torch.no_grad():
                new_embeds = model(x1_batch, output_hidden_states=True).hidden_states[0]
                h1_batch = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[args.layer_num + 1]
                z1_batch, _, _ = extract_sae_features(h1_batch[:, -1, :], sae, args.model_type, args.k)

            sorted_indices = torch.sort(z1_batch, dim=1, descending=True).indices
            rank_batch = (sorted_indices == n_id).nonzero(as_tuple=False)
            rank_per_sample = torch.full((z1_batch.shape[0],), z1_batch.shape[1], dtype=torch.long, device=z1_batch.device)
            rank_per_sample[rank_batch[:, 0]] = rank_batch[:, 1]
            best_idx = torch.argmin(rank_per_sample) if args.activate else torch.argmax(rank_per_sample)
            x1 = x1_batch[best_idx].unsqueeze(0)
            current_rank = rank_per_sample[best_idx].item()
            if (args.activate and current_rank < best_rank) or (not args.activate and current_rank > best_rank):
                best_rank = current_rank
            current_loss = torch.nn.functional.log_softmax(z1_batch[best_idx])[n_id].item()
            x1_text = tokenizer.decode(x1[0][:x1_init.shape[-1]], skip_special_tokens=True)
            print(f"Iteration {i+1} loss = {current_loss}")
            print(f"Iteration {i+1} best rank = {best_rank}")
            print(f"Iteration {i+1} input text: {x1_text}")
            if (args.activate and best_rank < len(s1)):
                print(f"Neuron {n_id} successfully activated!")
                print("--------------------")
                success_count += 1
                break
            if not args.activate and best_rank > len(s1):
                print(f"Neuron {n_id} successfully deactivated!")
                print("--------------------")
                success_count += 1
                break
        count += 1
        if args.activate and best_rank > len(s1):
            print(f"Neuron {n_id} cannot be activated")
            print("--------------------")
        if not args.activate and best_rank < len(s1):
            print(f"Neuron {n_id} cannot be deactivated")
            print("--------------------")

        all_final_ranks.append(best_rank)
        print(f"{success_count} out of {count} attacks are successful!")
        

    print(f"Successful rate = {success_count / len(neuron_list)}")
    print(f"All final ranks = {all_final_ranks}")
    sys.stdout.close()

def run_group_targeted_suffix_attack(args):
    df = pd.read_csv(args.data_file)
    sae = Sae.load_from_disk(args.base_dir + f"layers.{args.layer_num}").to(DEVICE)
    log_file_path = f"./results/llama3-8b-generated/layer-{args.layer_num}/targeted-group-suffix-{args.sample_idx}.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = open(log_file_path, "w")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
    model = LlamaForCausalLM.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    x1_raw_text = df.iloc[args.sample_idx]['x1'][:-1]
    x2_raw_text = df.iloc[args.sample_idx]['x2'][:-1]
    print(f"x1: {x1_raw_text}")
    print(f"x2: {x2_raw_text}")

    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x2_raw = tokenizer(x2_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)

    x1_raw_processed = tokenizer(x1_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
    x2_raw_processed = tokenizer(x2_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

    h1 = model(x1_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
    h2 = model(x2_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]

    z1 = sae.pre_acts(h1)
    z2 = sae.pre_acts(h2)
    s1 = sae.encode(h1).top_indices
    s2 = sae.encode(h2).top_indices

    target_diff = (z2 - z1).detach()

    best_loss = float('inf')
    x1_adv = x1_raw.clone()
    for i in range(args.num_iters):
        x1_adv = x1_adv.repeat(args.batch_size, 1)
        embeds = model.get_input_embeddings()(x1_adv).detach().requires_grad_(True)
        h1_batch = model(inputs_embeds=embeds, output_hidden_states=True).hidden_states[args.layer_num + 1][:, -1, :]
        z1_batch = sae.pre_acts(h1_batch)
        loss = -F.cosine_similarity(z1_batch, z2.unsqueeze(0)).mean()
        gradients = torch.autograd.grad(outputs=loss, inputs=embeds)[0]
        dot_prod = torch.matmul(gradients[:, -1, :], model.get_input_embeddings().weight.T)
        top_k_adv = torch.topk(dot_prod, args.k, dim=-1).indices

        # Apply sampled perturbation to the suffix token
        random_top = top_k_adv[torch.arange(args.batch_size), torch.randint(0, args.k, (args.batch_size,))]
        x1_adv[:, -1] = random_top

        with torch.no_grad():
            embeds_new = model.get_input_embeddings()(x1_adv)
            h1_batch_new = model(inputs_embeds=embeds_new, output_hidden_states=True).hidden_states[args.layer_num + 1][:, -1, :]
            z1_batch_new = sae.pre_acts(h1_batch_new)
            new_loss = -F.cosine_similarity(z1_batch_new, z2.unsqueeze(0)).mean()
            print(f"Iteration {i+1}, loss = {new_loss.item():.4f}")
            if new_loss < best_loss:
                best_loss = new_loss
                x1_best = x1_adv[0]

    print(f"Final loss = {best_loss:.4f}")
    print("Adversarial x1:")
    print(tokenizer.decode(x1_best, skip_special_tokens=True))
    if args.log:
        sys.stdout.close()

def run_group_untargeted_suffix_attack(args):
    df = pd.read_csv(args.data_file)
    sae = Sae.load_from_disk(args.base_dir + f"layers.{args.layer_num}").to(DEVICE)
    log_file_path = f"./results/llama3-8b-generated/layer-{args.layer_num}/untargeted-group-suffix-{args.sample_idx}.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = open(log_file_path, "w")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/")
    model = LlamaForCausalLM.from_pretrained(args.model_path, cache_dir="/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/").to(DEVICE)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    x1_raw_text = df.iloc[args.sample_idx]['x1'][:-1]
    print(f"x1: {x1_raw_text}")
    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x1_raw_processed = tokenizer(x1_raw_text + " \nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

    h1 = model(x1_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
    z1 = sae.pre_acts(h1)
    s1 = sae.encode(h1).top_indices

    for i in range(args.num_iters):
        x1_batch = x1_raw.repeat(args.batch_size, 1)
        embeds = model.get_input_embeddings()(x1_batch).detach().requires_grad_(True)
        h1_batch = model(inputs_embeds=embeds, output_hidden_states=True).hidden_states[args.layer_num + 1][:, -1, :]
        z1_batch = sae.pre_acts(h1_batch)

        cosine_to_original = F.cosine_similarity(z1_batch, z1.unsqueeze(0))
        loss = -cosine_to_original.mean()
        gradients = torch.autograd.grad(outputs=loss, inputs=embeds)[0]
        dot_prod = torch.matmul(gradients[:, -1, :], model.get_input_embeddings().weight.T)
        top_k_adv = torch.topk(dot_prod, args.k, dim=-1).indices

        sampled = top_k_adv[torch.arange(args.batch_size), torch.randint(0, args.k, (args.batch_size,))]
        x1_batch[:, -1] = sampled

        with torch.no_grad():
            new_embeds = model.get_input_embeddings()(x1_batch)
            h1_batch_new = model(inputs_embeds=new_embeds, output_hidden_states=True).hidden_states[args.layer_num + 1][:, -1, :]
            z1_batch_new = sae.pre_acts(h1_batch_new)
            new_cos_sim = F.cosine_similarity(z1_batch_new, z1.unsqueeze(0))
            print(f"Iteration {i+1}, cosine sim to original: {new_cos_sim.mean().item():.4f}")

    sys.stdout.close()
