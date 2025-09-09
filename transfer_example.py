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
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/"
CACHE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model_type = "gemma2-9b"
layer_num = 30
targeted = False
model, tokenizer, sae = load_model_and_sae(model_type, layer_num)

# df = pd.read_csv("./art_science.csv")
x1_text = "Her thesis analyzes representations of trauma and silence in Holocaust literature across multiple national contexts"
x2_text = ""
x1_adv_text = "Her thesis analyzes representations of trauma and silence in Holocaust literature across multiple national contexts妙"

x1_tokenized = tokenizer(x1_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
x2_tokenized = tokenizer(x2_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
x1_adv_tokenized = tokenizer(x1_adv_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

h1 = model(x1_tokenized, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
h2 = model(x2_tokenized, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]
h1_adv = model(x1_adv_tokenized, output_hidden_states=True).hidden_states[layer_num + 1][0][-1]

if targeted:
    z2, s2, _ = extract_sae_features(h2, sae, model_type)
    k = len(s2)
    z1, s1, _ = extract_sae_features(h1, sae, model_type, k)
else:
    z1, s1, _ = extract_sae_features(h1, sae, model_type)
    k = len(s1)

z1_adv, s1_adv, _ = extract_sae_features(h1_adv, sae, model_type, k)

if targeted:
    print(f"s1 s2 overlap = {count_common(s1, s2) / len(s2)} for {model_type}")
    print(f"s1_adv s2 overlap = {count_common(s1_adv, s2) / len(s2)} for {model_type}")
else:
    print(f"s1_adv s1 overlap = {count_common(s1_adv, s1) / len(s1)} for {model_type}")