import numpy
import torch
import torch.nn.functional as F

a = torch.tensor([
    [-1.0, 2.0],
    [-2.0, 3.0]
])

a.transpose(0,1)
print(a)
a.transpose(0,1)
print(a)
