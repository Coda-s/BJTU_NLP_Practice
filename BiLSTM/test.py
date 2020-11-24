import torch

def f():
    return 1, 2, 3

a = 2
b = 3
c = 4
a, b, c = f()
print(a,b,c)