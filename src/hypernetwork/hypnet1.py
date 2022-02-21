# Faisal Qureshi     
# faisal.qureshi@ontariotechu.ca

from re import I
import torch
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

device = 'cpu'

image_folder = '../../data/imagecompression.info/rgb8bit'
image_input = os.path.join(image_folder, 'deer-small.ppm')
image_output = os.path.join(image_folder, 'deer-small-enhanced.ppm')

def show_images(im1, im2, im3, delay=-1):
    """
    Uses matplotlib to show two images next to each other.
    """
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('input')
    plt.imshow(im1)
    plt.subplot(1,3,2)
    plt.title('network')
    plt.imshow(im2)
    plt.subplot(1,3,3)
    plt.title('corrected')
    plt.imshow(im3)

    if delay > 0:
        plt.pause(delay)
    else:
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
        # for p in self.backbone.parameters():
        #     p.requires_grad = False
        
        in_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_ftrs, 64)
        self.linear = torch.nn.Linear(64, 30, bias=False)
        self.ctx = None

    def forward(self):
        x = self.backbone(self.ctx)
        x = torch.nn.functional.relu(x)
        x = self.linear(x)
        return x.view(3,10)

class PrimaryNet(torch.nn.Module):
    """
    PrimaryNet inlucdes 10x3 matrix for going from r2,g2,b2,rg,rb,gb,r,g,b,1 space to r,g,b space.
    This matrix predicted by HyperNet.
    """

    def __init__(self):
        super(PrimaryNet, self).__init__()
        self.hyper_network = HyperNet()

    def forward(self, x):
        w = self.hyper_network()
        x = torch.nn.functional.linear(x, w)
        return x

def train(num_epochs, verbose, resume):
    print(f'Reading {image_input}')
    im_in = Image.open(image_input)
    print(im_in.size)
    print(f'Reading {image_output}')    
    im_out = Image.open(image_output)

    print('Setting up dataset')
    dataset = PixData(im_in, im_out)
    training_dataloader = tdata.DataLoader(dataset, batch_size=512, shuffle=True)

    net = PrimaryNet()

    # load previously saved checkpoint if resuming otherwise start anew
    if resume:
        ckpt = torch.load('./hyper1.pth')
        net.load_state_dict(ckpt['net'])
        loss_best = ckpt['loss']
        epochs_completed = ckpt['epochs_completed']

        print('Resuming from saved checkpoint:')
        print(f'  Epochs completed = {epochs_completed}, loss = {loss_best}')
    else:
        loss_best = -1
        epochs_completed = 0

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
        for _, data in enumerate(training_dataloader):
            inputs = data['data'].to(device)
            labels = data['out'].to(device)

            optimizer.zero_grad()
            net.hyper_network.ctx = dataset.ctx.to(device)
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

    im_corrected = evaluate(net, im_in)
    show_images(im_in, im_corrected, im_out)

    return net

def evaluate(net, im):
        transforms_1 = tv.transforms.Compose([tv.transforms.ToTensor()])
        input = lift(transforms_1(im))
        input = einops.rearrange(input, 'h w -> () w h')
        print(input.shape)
        transforms_2 = tv.transforms.Compose([tv.transforms.Resize((224,224)), tv.transforms.ToTensor()])
        ctx = einops.rearrange(transforms_2(im), 'c h w -> () c h w')
        print(ctx.shape)
        net.hyper_network.ctx = ctx.to(device)
        output = net(input.to(device)).view(3, 334, 512)
        print(output.shape)
        return einops.rearrange(output.detach().cpu(), 'c h w -> h w c')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', action='store_true', default=False, help='Resume training')
    parser.add_argument('-s', '--show', action='store_true', default=False, help='Show input and output images and exits')
    parser.add_argument('-e', '--num_epochs', action='store', default=1, type=int, help='Number of epochs')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print loss at every epoch')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # print(f'Reading {image_input}')
    # im_in = Image.open(image_input)
    # print(f'Reading {image_output}')    
    # im_out = Image.open(image_output)

    # if args.show:
    #     show_images(im_in, im_out)

    # print('Setting up dataset')
    # dataset = PixData(im_in, im_out)
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset.ctx.shape)

    # print('Setting up HyperNet')
    # hyper_net = HyperNet()
    # hyper_net(dataset.ctx)

    # print('Setting up PrimaryNet')
    # primary_net = PrimaryNet()
    # x = {'data': dataset[0]['dataz):'].unsqueeze(0), 'ctx': dataset.ctx}
    # primary_net(x)
    # print(primary_net.state_dict())
    
    net = train(num_epochs=args.num_epochs, verbose=args.verbose, resume=args.resume)

    