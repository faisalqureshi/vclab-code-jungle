# Faisal Qureshi  
# faisal.qureshi@ontariotechu.ca

import sys
sys.path.append('../')

import torch
import torch.utils.data as tdata
import positional_encoding as pe
import numpy as np

class PixelsDataset(tdata.Dataset):
    def __init__(self, image, pos_enc_dim=8, scale=1./255.):
        self.image = image * scale
        self.pos, self.enc_x, self.enc_y = self.construct_positional_encoding(self.image, pos_enc_dim=pos_enc_dim)
        self.n = len(self.pos)
        self.pos_enc_dim = pos_enc_dim
        self.scale = scale

    @staticmethod
    def construct_positional_encoding(image, pos_enc_dim):
        h, w = image.shape
        x, y = np.arange(w), np.arange(h)
        xx, yy = np.meshgrid(x, y)
        pos = np.stack([xx, yy, image], axis=2).reshape(-1, 3)
        enc_x = pe.positional_encoding(x, pos_enc_dim)
        enc_y = pe.positional_encoding(y, pos_enc_dim)
        return pos, enc_x, enc_y
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        x = self.pos[idx, 0]
        y = self.pos[idx, 1]
        e_x = self.enc_x[int(x)]
        e_y = self.enc_y[int(y)]
        return {
            'pos': torch.Tensor(np.hstack((e_x, e_y))),
            'c': torch.Tensor(self.pos[idx, 2:])
        }

