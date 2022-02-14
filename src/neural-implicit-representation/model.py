import torch

class Nir(torch.nn.Module):
    def __init__(self, h, w, pos_enc_dim, output_dim):
        super(Nir, self).__init__()
        
        self.linear1 = torch.nn.Linear(pos_enc_dim*2, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 16)        
        self.linear4 = torch.nn.Linear(16, output_dim)
        
        self.h, self.w, self.pos_enc_dim = h, w, pos_enc_dim
        
    def forward(self, x):
        x = torch.sin(self.linear1(x))
        x = torch.sin(self.linear2(x))
        x = torch.sin(self.linear3(x))
        x = self.linear4(x)
        return x