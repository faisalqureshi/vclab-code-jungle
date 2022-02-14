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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural implicit representation demo')
    parser.add_argument('--mnist-id', action='store', default=0, type=int, help='MNIST id of the image used for training')
    parser.add_argument('--data-dir', action='store', default='.', help='Directory containing data')
    parser.add_argument('--pos-enc-dim', action='store', type=int, default=8, help='Positional encoding dimenstion')
    parser.add_argument('--batchsize', action='store', type=int, default=16, help='Specify batchsize for training')
    parser.add_argument('--shuffle', action='store_true', default=False, help='Set shuffle to true.')
    parser.add_argument('--epochs', action='store', default=1, type=int, help='Number of epochs')


    args = parser.parse_args()

    image = pick_mnist_image(args.mnist_id, root=args.data_dir, download=False)
    h, w = image.shape

    plt.figure()
    plt.title(f'Input image. {h}x{w}. Press q to continue')
    plt.imshow(image, cmap='gray')
    plt.show()

    pixels_dataset = pix.PixelsDataset(image=image, pos_enc_dim=args.pos_enc_dim, scale=1./255.) 
    print(f'Number of pixels = {len(pixels_dataset)}')

    training_dataloader = DataLoader(pixels_dataset, batch_size=args.batchsize, shuffle=args.shuffle)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    nn = model.Nir(h, w, args.pos_enc_dim, 1)

    writer = SummaryWriter()

    nn.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):
        loss_epoch = 0.0
        cnts = 0
        for _, data in enumerate(training_dataloader):
            inputs = data['pos'].to(device)
            labels = data['c'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)    
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            cnts += 1
            loss_epoch += loss.cpu().item()
            
        loss_epoch /= cnts
        writer.add_scalar('Loss/train', loss, epoch)
            
        if epoch % 1000 == 0 or epoch == num_epochs-1:
            print(f'epoch = {epoch}: loss = {loss_epoch}')
            
    #torch.save(model.state_dict(), './nir.pts')


