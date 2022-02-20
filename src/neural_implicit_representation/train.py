# Faisal Qureshi  
# faisal.qureshi@ontariotechu.ca

import sys
import numpy as np
import torchvision as tv
import argparse
import matplotlib 
import matplotlib.pyplot as plt
import pixels as pix
import torch
from torch.utils.data import DataLoader
import model
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')

def pick_mnist_image(idx=0, root='../data', download=False):
    data = tv.datasets.MNIST(root=root, download=download)
    image = np.array(data[idx][0])
    return image

def train(mnist_id, data_dir, num_epochs, resume, verbose):
    # image
    image = pick_mnist_image(mnist_id, root=data_dir, download=False)
    h, w = image.shape

    # construct pixel dataset and dataloader, encoding_dimension
    pos_enc_dim = 8
    batch_size = 1024
    shuffle = True
    pixels_dataset = pix.PixelsDataset(image=image, pos_enc_dim=pos_enc_dim, scale=1./255.) 
    print(f'Number of pixels = {len(pixels_dataset)}')

    training_dataloader = DataLoader(pixels_dataset, batch_size=batch_size, shuffle=shuffle)

    # compute device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # model -- encoding dimension = 16, output 1     
    net = model.Nir(h, w, pos_enc_dim, 1)

    # load previously saved checkpoint if resuming otherwise start anew
    if resume:
        ckpt = torch.load('./nir.pth')
        net.load_state_dict(ckpt['net'])
        loss_best = ckpt['loss']
        epochs_completed = ckpt['epochs_completed']

        print('Resuming from saved checkpoint:')
        print(f'  Epochs completed = {epochs_completed}, loss = {loss_best}')
    else:
        loss_best = -1
        epochs_completed = 0

    # setting up tensorboard
    writer = SummaryWriter()

    # push model to compute device
    # set up loss and optimizer
    net.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Set up training loop
    print_every = 1 if verbose else 10

    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        loss_epoch = 0.0
        cnts = 0
        for _, data in enumerate(training_dataloader):
            inputs = data['pos'].to(device)
            labels = data['c'].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)    
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            cnts += 1
            loss_epoch += loss.cpu().item()
            
        loss_epoch /= cnts
        writer.add_scalar('Loss/train', loss, epoch+epochs_completed)
            
        if epoch == 1 or epoch % print_every == 0 or epoch == num_epochs-1:
            print(f'epoch = {epoch+epochs_completed}: loss = {loss_epoch}')
            
        if loss_best < 0 or loss_best > loss_epoch:
            torch.save(
                {
                    'net': net.state_dict(),
                    'loss': loss_epoch,
                    'epochs_completed': epoch+epochs_completed
                },
                './nir.pth'
            )
            loss_best = loss_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural implicit representation demo')
    parser.add_argument('mnistid', action='store', type=int, help='MNIST id of the image used for training')
    parser.add_argument('--data-dir', action='store', default='.', help='Directory containing data')
    parser.add_argument('--num_epochs', action='store', default=1, type=int, help='Number of epochs')
    parser.add_argument('-r', '--resume', action='store_true', default=False, help='Resume training')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print loss at every epoch')

    # for argparse it is best to not use das
    
    args = parser.parse_args()

    # plt.figure()
    # plt.title(f'Input image. {h}x{w}. Press q to continue')
    # plt.imshow(image, cmap='gray')
    # plt.show()

    train(args.mnistid, args.data_dir, args.num_epochs, args.resume, args.verbose)



