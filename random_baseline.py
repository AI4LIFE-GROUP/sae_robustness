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