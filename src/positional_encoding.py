import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def positional_encoding(pos, d_model, min_freq=1/10000.0):
    powers = 2*(np.arange(d_model)//2)/d_model
    freqs = min_freq**powers
    enc = pos.reshape(-1,1)*freqs.reshape(1,-1)
    enc[:,0::2] = np.sin(enc[:,0::2])
    enc[:,1::2] = np.cos(enc[:,1::2])
    return enc

if __name__ == '__main__':
    n = 64
    d_model = 64
    enc = positional_encoding(np.arange(n), d_model)
    print(enc)