import numpy as np
import torch
import matplotlib.pyplot as plt
from sae import Sae

saes = Sae.load_many_from_hub("EleutherAI/sae-llama-3-8b-32x")

