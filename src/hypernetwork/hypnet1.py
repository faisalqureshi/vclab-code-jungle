# Faisal Qureshi     
# faisal.qureshi@ontariotechu.ca

import torch
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

image_folder = '../../data/imagecompression.info/rgb8bit'
image_input = os.path.join(image_folder, 'deer-small.ppm')
image_output = os.path.join(image_folder, 'deer-small-enhanced.ppm')

def show_images(im1, im2):
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('original')
    plt.imshow(im1)
    plt.subplot(1,2,2)
    plt.title('enhanced')
    plt.imshow(im2)
    plt.show()

import einops
def lift(im):
    """lifts input (r,g,b) image to (r2,g2,b2,rg,rb,gb,r,g,b,1)"""
    _, h, w = im.shape
    x = einops.rearrange(im, 'c h w -> c (h w)')
    x = torch.vstack((x**2, x[0,:]*x[1,:], x[0,:]*x[2,:], x[1,:]*x[2,:], x, torch.ones(h*w)))
    return x

import torch.utils.data as tdata
class PixData(tdata.Dataset):
    def __init__(self, image_in, image_out):
        """Input and output images 8bit rgb images."""
        transforms_1 = tv.transforms.Compose([tv.transforms.ToTensor()])
        self.image_in = lift(transforms_1(image_in))
        self.image_out = einops.rearrange(transforms_1(image_out), 'c h w -> c (h w)')

        transforms_2 = tv.transforms.Compose([tv.transforms.Resize((224,224)), tv.transforms.ToTensor()])
        self.ctx = einops.rearrange(transforms_2(image_in), 'c h w -> () c h w')

        _, self.len = self.image_in.shape
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return {
            'data': self.image_in[:, idx],
            'out': self.image_out[:, idx]
        }


class HyperNet(torch.nn.Module):
    """
    HyperNet predicts a 30 dimensional tensor that sets the matrix for the PrimaryNet shown
    below.  
    """

    def __init__(self):
        super(HyperNet, self).__init__()
        self.backbone = tv.models.resnet18(pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        in_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_ftrs, 30, bias=False)

    def forward(self, ctx):
        x = self.backbone(ctx)
        return x

class PrimaryNet(torch.nn.Module):
    """
    PrimaryNet inlucdes 10x3 matrix for going from r2,g2,b2,rg,rb,gb,r,g,b,1 space to r,g,b space.
    This matrix predicted by HyperNet.
    """

    def __init__(self):
        super(PrimaryNet, self).__init__()
        self.linear = torch.nn.Linear(10, 3, bias=False)
        self.hyper_network = HyperNet()

    def forward(self, x):
        weights = self.hyper_network(x['ctx'])
        self.linear.weight = torch.nn.Parameter(weights.view(3,10))
        self.linear.weight.requires_grad = False
        x = self.linear(x['data'])
        return x

def train(num_epochs, verbose, resume):
    print(f'Reading {image_input}')
    im_in = Image.open(image_input)
    print(f'Reading {image_output}')    
    im_out = Image.open(image_output)

    print('Setting up dataset')
    dataset = PixData(im_in, im_out)
    training_dataloader = tdata.DataLoader(dataset, batch_size=512, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    net = PrimaryNet()

    # push model to computed device
    # set up loss and optimizer
    net.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.001, momentum=0.9)

    writer = SummaryWriter()

    print_every = 1 if verbose else 10

    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        loss_epoch = 0.0
        cnts = 0
        for _, x in enumerate(training_dataloader):
            inputs = {
                'data': x['data'].to(device),
                'ctx': dataset.ctx.to(device)
            }
            labels = x['out'].to(device)

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
                './hyper1.pth'
            )
            loss_best = loss_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', action='store_true', default=False, help='Resume training')
    parser.add_argument('-s', '--show', action='store_true', default=False, help='Show input and output images and exits')
    parser.add_argument('--num_epochs', action='store', default=1, type=int, help='Number of epochs')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print loss at every epoch')

    args = parser.parse_args()

    print(f'Reading {image_input}')
    im_in = Image.open(image_input)
    print(f'Reading {image_output}')    
    im_out = Image.open(image_output)

    if args.show:
        show_images(im_in, im_out)

    print('Setting up dataset')
    dataset = PixData(im_in, im_out)
    print(len(dataset))
    print(dataset[0])
    print(dataset.ctx.shape)

    print('Setting up HyperNet')
    hyper_net = HyperNet()
    hyper_net(dataset.ctx)

    print('Setting up PrimaryNet')
    primary_net = PrimaryNet()
    x = {'data': dataset[0]['data'].unsqueeze(0), 'ctx': dataset.ctx}
    primary_net(x)
    print(primary_net.state_dict())
    
    train(num_epochs=args.num_epochs, verbose=args.verbose, resume=args.resume)