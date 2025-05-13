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
os.environ["CUDA_VISIBLE_DEVICES"] = ""
BASE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/sae/"
CACHE_DIR = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model, tokenizer, sae = load_model_and_sae("llama3-8b", 20)
df = pd.read_csv("./two_class_generated.csv")

gen_len = 30

x1_raw_text = "Philosophers from Plato to Sartre have long debated whether human choices are governed by free will or determined forces"
x1_adv = "Philosophers from Plato to Sartre have long debated whether human choices are governed by free will or determined forces.Components microscopy​​"

x1_tokenized = tokenizer(x1_raw_text + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)
x1_adv_tokenized = tokenizer(x1_adv + "\nThe previous text is about", return_tensors="pt")['input_ids'].to(DEVICE)

x1_out = model.generate(x1_tokenized, max_length=x1_tokenized.shape[-1] + gen_len, do_sample=False, num_return_sequences=1)
x1_adv_out = model.generate(x1_adv_tokenized, max_length=x1_adv_tokenized.shape[-1] + gen_len, do_sample=False, num_return_sequences=1)

x1_generated = tokenizer.decode(x1_out[0][x1_tokenized.shape[-1]:], skip_special_tokens=True)
x1_adv_generated = tokenizer.decode(x1_adv_out[0][x1_adv_tokenized.shape[-1]:], skip_special_tokens=True)

print(f"x1: {x1_generated}")
print(f"x1_adv: {x1_adv_generated}")

checkpoint = "/n/netscratch/hlakkaraju_lab/Lab/aaronli/models/BLEURT-20"
references = [x1_generated]
candidates = [x1_adv_generated]

scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=references, candidates=candidates)

print(scores)