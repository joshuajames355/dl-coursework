# this code is based on [ref], which is released under the MIT licesne
# make sure you reference any code you have studied as above here

# imports
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from models import VAE
import visdom

vis = visdom.Visdom(port=12345, env="vae")

# hyperparameters
batch_size  = 64
n_channels  = 3
latent_size = 256
beta = 10
dataset = 'cifar10'
fileName = "vae_checkpoint.chkpt"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#filters only elements with the corresponding labels
def cifarFilter(dataset, labels=[2,7]):
    idx = list(filter(lambda x: dataset.targets[x] in labels ,range(len(dataset))))
    dataset.targets = [dataset.targets[x] for x in idx]
    dataset.data = [dataset.data[x] for x in idx]  
    return dataset 

# you may use cifar10 or stl10 datasets
if dataset == 'cifar10':
    dataset = torchvision.datasets.CIFAR10('cifar10', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))

    dataset = cifarFilter(dataset)  
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# stl10 has larger images which are much slower to train on. You should develop your method with CIFAR-10 before experimenting with STL-10
if dataset == 'stl10':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.STL10('drive/My Drive/training/stl10', split='train+unlabeled', download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])),
    shuffle=True, batch_size=batch_size, drop_last=True)
    train_iterator = iter(cycle(train_loader))
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'] # these are slightly different to CIFAR-10

train_iterator = iter(cycle(train_loader))


A = VAE(n_channels, latent_size).to(device)
print(f'> Number of autoencoder parameters {len(torch.nn.utils.parameters_to_vector(A.parameters()))}')
optimiser = torch.optim.Adam(A.parameters(), lr=0.001)
epoch = 0

vis.line(X=[0], Y=[0], win="Loss", opts=dict(title="Loss"), )

# training loop, you will want to train for more than 10 here!
while (epoch<10000):
    
    # array(s) for the performance measures
    loss_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(100):

        # sample x from the dataset
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)
        #print(t)

        # do the forward pass with mean squared error
        x_hat, mu, logvar = A(x)

        #losses are summed over the batch
        reconstruction_loss = ((x-x_hat)**2).sum()
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + beta * kl_loss

        # backpropagate to compute the gradient of the loss w.r.t the parameters and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # collect stats
        loss_arr = np.append(loss_arr, loss.item())

    # sample your model (autoencoders are not good at this)
    #z = torch.randn_like(z)
    #g = A.decode(z)

    # plot some examples
    vis.line(X=[epoch], Y=[loss.mean().item()], win="Loss", opts=dict(title="Loss"), update="append")

    #plt.rcParams['figure.dpi'] = 100
    #plt.grid(False)
    #plt.imshow(torchvision.utils.make_grid(g[:8]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    #plt.show()
    #plt.pause(0.0001)

    epoch = epoch+1

    torch.save({'A':A.state_dict(), 'optimiser':optimiser.state_dict(), 'epoch':epoch}, 'save.chkpt')