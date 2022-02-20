# Faisal Qureshi  
# faisal.qureshi@ontariotechu.ca

import torch

class Sine(torch.nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Nir(torch.nn.Module):
    def __init__(self, h, w, pos_enc_dim, output_dim):
        super(Nir, self).__init__()
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.linear1 = torch.nn.Linear(pos_enc_dim*2, 32)
        self.sin1 = Sine()
        self.linear2 = torch.nn.Linear(32, 32)
        self.sin2 = Sine()
        self.linear3 = torch.nn.Linear(32, 16)
        self.sin3 = Sine()
        self.linear4 = torch.nn.Linear(16, output_dim)
        
        self.h, self.w, self.pos_enc_dim = h, w, pos_enc_dim
        
    def forward(self, x):
        #x = self.quant(x)
        
        x = self.linear1(x)
        x = self.sin1(x)
        x = self.linear2(x)
        x = self.sin2(x)
        x = self.linear3(x)
        x = self.sin3(x)
        x = self.linear4(x)
        
        #x = self.dequant(x)
        
        return x