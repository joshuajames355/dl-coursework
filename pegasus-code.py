#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#https://github.com/pytorch/examples/tree/master/dcgan

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
from models import Generator, Discriminator
import visdom
from torchsummary import summary


# hyperparameters
batch_size  = 64
n_channels  = 3
latent_size = 256
beta = 10
fileName = "dcgan_checkpoint.chkpt"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WHITE_BONUS_FACTOR = 0.1
RESUME_TRAINING = True
ENABLE_VISDOM = False
VISDOM_PORT = 12345

if ENABLE_VISDOM:
    vis = visdom.Visdom(port=12345, env="dcgan")




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

dataset = torchvision.datasets.CIFAR10('cifar10', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ]))

dataset = cifarFilter(dataset)  
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_iterator = iter(cycle(train_loader))

G = Generator().to(device)
D = Discriminator().to(device)

print(f'> Number of Generator parameters {len(torch.nn.utils.parameters_to_vector(G.parameters()))}')

g_optimiser = torch.optim.Adam(G.parameters(), lr=0.001)
d_optimiser = torch.optim.Adam(D.parameters(), lr=0.001) #discriminator bird

epoch = 0

if RESUME_TRAINING:
    params = torch.load(fileName)
    G.load_state_dict(params["G"])
    g_optimiser.load_state_dict(params["G_optimiser"])
    epoch = int(params["epoch"])
    D.load_state_dict(params["D"])
    d_optimiser.load_state_dict(params["D_optimiser"])
    print("Loaded from: " + fileName)

criterion = nn.BCELoss()

if ENABLE_VISDOM:
    vis.line(X=[0], Y=[0], win="GLoss", opts=dict(title="Generator Loss Combined"), )
    vis.line(X=[0], Y=[0], win="DLoss", opts=dict(title="Discriminator Loss"), )

white = torch.full((3,32,32),1, device=device)

mse = nn.MSELoss()
sig = torch.nn.Sigmoid()   

def whiteBonus(x):
    loss = torch.tensor([mse(i, white) for i in x], device=device)
    return sig(loss)    

#training loop, you will want to train for more than 10 here!
while (epoch<10000):
    # array(s) for the performance measures
    loss_arr_g = np.zeros(0)
    loss_arr_d = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(100):

        # sample x from the dataset
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        #reset optimizers
        d_optimiser.zero_grad()
        g_optimiser.zero_grad()

        fake_label = torch.tensor([[0, 0] for _ in t], device=device, dtype=torch.float)
        real_label = torch.tensor([[1, 1] for _ in t], device=device, dtype=torch.float)

        #[x][0] for bird data
        #[x][1] for horse data
        training_labels = torch.tensor([[0, 1] if i == 7 else [1, 0] for i in t], device=device, dtype=torch.float)

        out = D(x) #train discriminator Bird on real data
        lossRD = criterion(out, training_labels)
        lossRD.backward()

        #train discriminator Bird on fake data
        noise = torch.randn(batch_size, 128, 1, 1).to(device)
        fakeData = G(noise)
        out2 = D(fakeData.detach())
        lossFD = criterion(out2, fake_label)
        lossFD.backward()
        d_optimiser.step()

        #collect discriminator stats
        lossDB = lossFD.cpu().item() + lossRD.cpu().item()
        loss_arr_d = np.append(loss_arr_d, lossDB)

        #train generator on fake data bird
        out3 = D(fakeData) #redo after the optimizer step
        lossFG = (out3 - real_label).pow(2).mean() + WHITE_BONUS_FACTOR*whiteBonus(fakeData).mean()
        lossFG.backward()
        g_optimiser.step()


        #collect generator stats
        loss_arr_g = np.append(loss_arr_g, lossFG.cpu().item() )

    if ENABLE_VISDOM:
        vis.line(X=[epoch], Y=[loss_arr_g.mean().item()], win="GLoss", opts=dict(title="Generator Loss"), update="append")
        vis.line(X=[epoch], Y=[loss_arr_d.mean().item()], win="DLoss", opts=dict(title="Discriminator Loss"), update="append")

        #eval print 5x5 grid
        noise = torch.randn(batch_size, 128, 1, 1).to(device)
        fakeData = G(noise)
        t = torchvision.utils.make_grid(fakeData)
        vis.image(t, win="image", opts=dict(title="image"))

    epoch += 1
    torch.save({'G':G.state_dict(), 
        'G_optimiser': g_optimiser.state_dict(), 
        'epoch':epoch,
        'D':D.state_dict(), 
        'D_optimiser': d_optimiser.state_dict(),}, fileName)


