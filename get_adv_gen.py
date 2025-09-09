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
from bleurt import score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/"
CACHE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model, tokenizer, sae = load_model_and_sae("llama3-8b", 20)
df = pd.read_csv("./two_class_generated.csv")
sample_idx = 20
gen_len = 30

x1_raw_text = "Gothic cathedrals like Notre-Dame embody religious fervor, architectural innovation, and medieval craftsmanship through pointed arches and stained glass"
# x2_raw_text = df.iloc[sample_idx]['x2'][:-1]
x1_adv = "Gothic cathedrals like Notre-Dame embody religious fervor, architectural innovation, and medieval craftsmanship through pointed arches and stained glassuncture.The fungi"

x1_tokenized = tokenizer(x1_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
# x2_tokenized = tokenizer(x2_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
x1_adv_tokenized = tokenizer(x1_adv + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

x1_out = model.generate(x1_tokenized, max_length=x1_tokenized.shape[-1] + gen_len, do_sample=False, num_return_sequences=1)
# x2_out = model.generate(x2_tokenized, max_length=x2_tokenized.shape[-1] + gen_len, do_sample=False, num_return_sequences=1)
x1_adv_out = model.generate(x1_adv_tokenized, max_length=x1_adv_tokenized.shape[-1] + gen_len, do_sample=False, num_return_sequences=1)

x1_generated = tokenizer.decode(x1_out[0][x1_tokenized.shape[-1]:], skip_special_tokens=True)
# x2_generated = tokenizer.decode(x2_out[0][x2_tokenized.shape[-1]:], skip_special_tokens=True)
x1_adv_generated = tokenizer.decode(x1_adv_out[0][x1_adv_tokenized.shape[-1]:], skip_special_tokens=True)

print(f"x1: {x1_generated}")
# print(f"x2: {x2_generated}")
print(f"x1_adv: {x1_adv_generated}")