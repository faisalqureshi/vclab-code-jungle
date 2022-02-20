# Similarity functions for comparing images.
# 
# Faisal Qureshi  
# faisal.qureshi@ontariotechu.ca

import torch

def mse(i1, i2):
    """
    Computes mse between two torch tensors
    """
    h, w = i1.shape
    return torch.sum((i1 - i2)**2) / (h*w)

def psnr(i1, i2, r=1.):
    """
    Computes psnr betweent two torch tensors
    """
    return 10*torch.log10(r**2 / mse(i1, i2))

def normalize(i):
    i -= i.min()
    i /= i.max()
    return i