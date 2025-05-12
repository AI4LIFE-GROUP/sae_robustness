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

def get_overlap(s_batch, s_ref):
    """
    Returns fraction of elements in s_ref that are also in s_batch.

    - s_batch: (k,) or (B, k)
    - s_ref: (k,)
    """
    if s_batch.dim() == 1:
        return (torch.isin(s_ref, s_batch).sum().float() / s_ref.numel())

    elif s_batch.dim() == 2:
        ref_set = set(s_ref.tolist())
        overlaps = []
        for row in s_batch:
            overlap = len(set(row.tolist()) & ref_set) / len(ref_set)
            overlaps.append(overlap)
        return torch.tensor(overlaps, device=s_batch.device)

    else:
        raise ValueError("s_batch must be 1D or 2D tensor")

def run_individual_suffix_attack(args):
    df = pd.read_csv(args.data_file)
    model, tokenizer, sae = load_model_and_sae(args.model_type, args.layer_num)
    if args.log:
        log_file_path = f"./results/{args.model_type}/layer-{args.layer_num}/{args.data_file}/{args.targeted}-individual-suffix-{args.sample_idx}-{'activate' if args.activate else 'deactivate'}.txt"
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
            mask = ~torch.isin(s2, s1_raw)
            top_indices = torch.topk(s2_acts[mask], k=min(args.num_latents, s2_acts[mask].numel())).indices
            neuron_list = s2[mask][top_indices]
        else:
            # Neurons in s1 but not in s2
            mask = ~torch.isin(s1_raw, s2)
            top_indices = torch.topk(s1_acts_raw[mask], k=min(args.num_latents, s1_acts_raw[mask].numel())).indices
            neuron_list = s1_raw[mask][top_indices]
    else:
        print(f"Initial s1_raw s1 overlap = {count_common(s1_raw, s1) / len(s1_raw)}")
        neuron_list = torch.topk(-z1_signed, k=args.num_latents).indices if args.activate else s1_raw[torch.topk(s1_acts_raw, k=args.num_latents).indices]
    
    success_count, count = 0, 0
    all_final_ranks = []
    
    for n_id in neuron_list:
        x1 = x1_init_processed.clone()
        best_rank = z1.shape[-1] if args.activate else 0

        for i in range(args.num_iters):
            with torch.no_grad():
                embeddings = model.get_input_embeddings()(x1) 
            embeddings = embeddings.detach().clone().requires_grad_(True)
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
                h1_batch = model(x1_batch, output_hidden_states=True).hidden_states[args.layer_num + 1]
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
            # print((z1_batch[best_idx] > 0.0).sum())
            if args.activate and z1_batch[best_idx][n_id] > 0 and args.model_type == "gemma2-9b":
                print(f"Neuron {n_id} successfully activated!")
                print("--------------------")
                success_count += 1
                break
            if args.activate and best_rank <= len(s1) and args.model_type == "llama3-8b":
                print(f"Neuron {n_id} successfully activated!")
                print("--------------------")
                success_count += 1
                break
            if not args.activate and z1_batch[best_idx][n_id] <= 0 and args.model_type == "gemma2-9b":
                print(f"Neuron {n_id} successfully deactivated!")
                print("--------------------")
                success_count += 1
                break
            if not args.activate and best_rank > len(s1) and args.model_type == "llama3-8b":
                print(f"Neuron {n_id} successfully deactivated!")
                print("--------------------")
                success_count += 1
                break
        count += 1
        if args.activate and z1_batch[best_idx][n_id] <= 0 and args.model_type == "gemma2-9b":
            print(f"Neuron {n_id} cannot be activated")
            print("--------------------")
        if args.activate and best_rank > len(s1) and args.model_type == "llama3-8b":
            print(f"Neuron {n_id} cannot be activated")
            print("--------------------")
        if not args.activate and z1_batch[best_idx][n_id] > 0 and args.model_type == "gemma2-9b":
            print(f"Neuron {n_id} cannot be deactivated")
            print("--------------------")
        if not args.activate and best_rank <= len(s1) and args.model_type == "llama3-8b":
            print(f"Neuron {n_id} cannot be activated")
            print("--------------------")

        all_final_ranks.append(best_rank)
        print(f"{success_count} out of {count} attacks are successful!")
        
    success_rate = success_count / len(neuron_list)
    print(f"Successful rate = {success_count / len(neuron_list)}")
    print(f"All final ranks = {all_final_ranks}")
    
    return success_rate

def run_population_suffix_attack(args):
    df = pd.read_csv(args.data_file)
    model, tokenizer, sae = load_model_and_sae(args.model_type, args.layer_num)
    if args.log:
        log_file_path = f"./results/{args.model_type}/layer-{args.layer_num}/{args.data_file}/{args.targeted}-population-suffix-{args.sample_idx}.txt"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        sys.stdout = open(log_file_path, "w")

    x1_raw_text = df.iloc[args.sample_idx]['x1'][:-1]
    print(f"x1: {x1_raw_text}")
    x1_raw = tokenizer(x1_raw_text, return_tensors="pt")['input_ids'].to(DEVICE)
    x1_raw_processed = tokenizer(x1_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
    h1_raw = model(x1_raw_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1].detach()

    x1_init = model.generate(x1_raw, max_length=x1_raw.shape[-1] + args.suffix_len, do_sample=False, num_return_sequences=1)
    x1_init_text = tokenizer.decode(x1_init[0], skip_special_tokens=True)
    print(f"x1 init: {x1_raw_text}")
    x1_init_processed = tokenizer(x1_init_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
    h1_init = model(x1_init_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1].detach()

    if args.targeted:
        x2_text = df.iloc[args.sample_idx]['x2'][:-1]
        print(f"x2: {x2_text}")
        x2 = tokenizer(x2_text, return_tensors="pt")['input_ids'].to(DEVICE)
        x2_processed = tokenizer(x2_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
        h2 = model(x2_processed, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1].detach()
        z2, s2, s2_acts = extract_sae_features(h2, sae, args.model_type, k=None)
        k = len(s2)
        z1_raw, s1_raw, s1_acts_raw = extract_sae_features(h1_raw, sae, args.model_type, k)
        z1_init, s1_init, s1_acts_init = extract_sae_features(h1_init, sae, args.model_type, k)
        initial_overlap = get_overlap(s1_raw, s2).item()
        print(f"Initial s1_raw s2 overlap = {initial_overlap}")
        print(f"Initial s1_init s2 overlap = {get_overlap(s1_init, s2)}")
    else:
        z1_raw, s1_raw, s1_acts_raw = extract_sae_features(h1_raw, sae, args.model_type, k=None)
        k = len(s1_raw)
        z1_init, s1_init, s1_acts_init = extract_sae_features(h1_init, sae, args.model_type, k)
        initial_overlap = 1.0
        print(f"Initial s1_raw s1_init overlap = {get_overlap(s1_raw, s1_init)}")

    best_overlap = 0.0 if args.targeted else 1.0
    x1 = x1_init_processed.clone()
    losses = []
    overlaps = []
    for i in range(args.num_iters):
        with torch.no_grad():
            embeddings = model.get_input_embeddings()(x1) 
        embeddings = embeddings.detach().clone().requires_grad_(True)
        h1 = model(inputs_embeds=embeddings, output_hidden_states=True).hidden_states[args.layer_num + 1][0][-1]
        z1, s1, s1_acts = extract_sae_features(h1, sae, args.model_type, k)
        if args.targeted:
            loss = F.cosine_similarity(z1, z2, dim=0)
        else:
            loss = -F.cosine_similarity(z1, z1_raw, dim=0)
        gradients = torch.autograd.grad(outputs=loss, inputs=embeddings, create_graph=False)[0]
        del h1, z1, s1, s1_acts
        torch.cuda.empty_cache()
        dot_prod = torch.matmul(gradients[0], model.get_input_embeddings().weight.T)
        dot_prod[:, tokenizer.eos_token_id] = -float('inf')
        top_m_adv = (torch.topk(dot_prod, args.m).indices)[x1_init.shape[-1] - args.suffix_len:x1_init.shape[-1]]

        x1_batch = x1.repeat(args.batch_size, 1).clone()
        rand_token_idx = torch.randint(0, args.suffix_len, (args.batch_size,))
        rand_top_m_idx = torch.randint(0, args.m, (args.batch_size,))
        batch_indices = torch.arange(args.batch_size)
        x1_batch[:, x1_init.shape[-1] - args.suffix_len:x1_init.shape[-1]][batch_indices, rand_token_idx] = top_m_adv[0, rand_top_m_idx]
        
        with torch.no_grad():
            h1_batch = model(x1_batch, output_hidden_states=True).hidden_states[args.layer_num + 1]
            z1_batch, s1_batch, s1_acts_batch = extract_sae_features(h1_batch[:, -1, :], sae, args.model_type, k)
        
        if args.targeted:
            overlap_batch = get_overlap(s1_batch, s2)
            best_idx = torch.argmax(overlap_batch)
        else:
            overlap_batch = get_overlap(s1_batch, s1_raw)
            best_idx = torch.argmin(overlap_batch)

        x1 = x1_batch[best_idx].unsqueeze(0)
        if (args.targeted and overlap_batch[best_idx] > best_overlap) or (not args.targeted and overlap_batch[best_idx] < best_overlap):
            best_overlap = overlap_batch[best_idx].item()
            best_x1 = x1[0][:x1_init.shape[-1]]
        if args.targeted:
            current_loss = F.cosine_similarity(z1_batch[best_idx], z2, dim=0).item()
        else:
            current_loss = F.cosine_similarity(z1_batch[best_idx], z1_raw, dim=0).item()

        x1_text = tokenizer.decode(best_x1, skip_special_tokens=True)
        losses.append(current_loss)
        overlaps.append(best_overlap)
        
        print(f"Iteration {i+1} loss = {current_loss}")
        print(f"Iteration {i+1} best overlap ratio = {best_overlap}")
        print(f"Iteration {i+1} input text: {x1_text}")
        print("--------------------")

        del h1_batch, z1_batch, s1_batch, s1_acts_batch
        torch.cuda.empty_cache()

    print(f"Final loss = {current_loss}")
    print(f"Final overlap = {best_overlap}")
    print(f"Final x1: {x1_text}")
    # if args.log:
    #     sys.stdout.close()

    return (best_overlap - initial_overlap) / initial_overlap