import numpy as np
import torch
import matplotlib.pyplot as plt



a = torch.tensor([1, -2, -3, 4, 5, 6])
b = torch.tensor([2, 4, 6, -8, 10, 12])

num_agreements = torch.sum(torch.sign(a) == torch.sign(b))
print(num_agreements)
# print((torch.sign(a) == torch.sign(b)) == )