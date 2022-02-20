# Faisal Qureshi  
# faisal.qureshi@ontariotechu.ca

import sys
sys.path.append('../')

import numpy as np
import argparse
import torch
import model
import positional_encoding as pe

def reconstruct_image(model, device=None):
    if device == None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    model.to(device)

    h, w, pos_enc_dim = model.h, model.w, model.pos_enc_dim
    
    output_image = np.empty((h,w))
    x, y = np.arange(w), np.arange(h)
    xx, yy = np.meshgrid(x, y)
    pos = np.stack([xx, yy], axis=2).reshape(-1, 2)

    model.eval()
    for i in range(pos.shape[0]):
        x, y = pos[i, 0], pos[i, 1]
        enc_x = pe.positional_encoding(x, pos_enc_dim)
        enc_y = pe.positional_encoding(y, pos_enc_dim)
        inputs = torch.Tensor(np.hstack((enc_x, enc_y))).unsqueeze(0).to(device)
        c = model(inputs)
        output_image[int(y)][int(x)] = c.detach().cpu()[0][0][0].numpy()
    
    return output_image

def resample_image(model, new_h, new_w, device=None):
    if device == None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    model.to(device)

    h, w, pos_enc_dim = model.h, model.w, model.pos_enc_dim
    
    output_image = np.empty((new_h,new_w))
    x, y = np.linspace(0, w, new_w), np.linspace(0, h, new_h)    
    xx, yy = np.meshgrid(x, y)
    xx_, yy_ = np.meshgrid(np.arange(new_w), np.arange(new_h))
    pos = np.stack([xx, yy, xx_, yy_], axis=2).reshape(-1, 4)

    model.eval()
    for i in range(pos.shape[0]):
        x, y, x_, y_ = pos[i, 0], pos[i, 1], int(pos[i, 2]), int(pos[i, 3])
        enc_x = pe.positional_encoding(x, pos_enc_dim)
        enc_y = pe.positional_encoding(y, pos_enc_dim)
        inputs = torch.Tensor(np.hstack((enc_x, enc_y))).unsqueeze(0).to(device)
        c = model(inputs)        
        output_image[y_][x_] = c.detach().cpu()[0][0][0].numpy()
    
    return output_image   

def load_model(filepath):
    # model -- important that his model matches the checkpoint
    # file     
    h, w = 28, 28
    pos_enc_dim = 8
    net = model.Nir(h, w, pos_enc_dim, 1)

    if filepath:
        ckpt = torch.load(filepath)
        net.load_state_dict(ckpt['net'])
        loss_best = ckpt['loss']
        epochs_completed = ckpt['epochs_completed']

        print('Using checkpoint:')
        print(f'  Epochs completed = {epochs_completed}, loss = {loss_best}')  

    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural implicit representation demo')
    parser = argparse.add_argument('-f', '--file', action='store', dafault=None, help='Checkpoint file')
    parser = argparse.add_argument('-u', '--upsample', action='store_true', default=False, help='Upsample if specified. Otherwise reconstruct the mnist digit.')
    args = parser.parse_args()

    model = load_model(args.file)
    if args.upsample:
        resample_image(model, 128, 128)
    else:
        reconstruct_image(model)