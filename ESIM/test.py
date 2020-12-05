import numpy
import torch
import torch.nn.functional as F

a = torch.tensor([
    [-1.0, 2.0],
    [-2.0, 3.0]
])

b = torch.tensor([3, 2])

print((torch.max(a, 1)[1] == b).sum())