#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

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

vis = visdom.Visdom(port=12345, env="dcgan")

# hyperparameters
batch_size  = 64
n_channels  = 3
latent_size = 256
beta = 10
dataset = 'cifar10'
fileName = "dcgan_checkpoint.chkpt"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
RESUME_TRAINING = False

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


G = Generator().to(device)
DB = Discriminator().to(device)
DH = Discriminator().to(device)


print(f'> Number of Generator parameters {len(torch.nn.utils.parameters_to_vector(G.parameters()))}')

g_optimiser = torch.optim.Adam(G.parameters(), lr=0.001)
db_optimiser = torch.optim.Adam(DB.parameters(), lr=0.001) #discriminator bird
dh_optimiser = torch.optim.Adam(DH.parameters(), lr=0.001) #discriminator horse

epoch = 0

if RESUME_TRAINING:
    params = torch.load(fileName)
    G.load_state_dict(params["G"])
    g_optimiser.load_state_dict(params["G_optimiser"])
    epoch = int(params["epoch"])
    DB.load_state_dict(params["DB"])
    db_optimiser.load_state_dict(params["DB_optimiser"])
    DH.load_state_dict(params["DH"])
    dh_optimiser.load_state_dict(params["DH_optimiser"])
    print("Loaded from: " + fileName)

criterion = nn.BCELoss()

vis.line(X=[0], Y=[0], win="GLoss", opts=dict(title="Generator Loss Combined"), )
vis.line(X=[0], Y=[0], win="DBLoss", opts=dict(title="Discriminator Bird Loss"), )
vis.line(X=[0], Y=[0], win="DHLoss", opts=dict(title="Discriminator Horse Loss"), )

#training loop, you will want to train for more than 10 here!
while (epoch<10000):
    # array(s) for the performance measures
    loss_arr_g = np.zeros(0)
    loss_arr_db = np.zeros(0)
    loss_arr_dh = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(100):

        # sample x from the dataset
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        #reset optimizers
        db_optimiser.zero_grad()
        dh_optimiser.zero_grad()
        g_optimiser.zero_grad()

        real_label = torch.full((batch_size,), 1, device=device)
        fake_label = torch.full((batch_size,), 0, device=device)
        horse_label = torch.tensor([i == 7 for i in t], device=device, dtype=torch.float)
        bird_label = torch.tensor([i == 2 for i in t], device=device, dtype=torch.float)

        out = DB(x) #train discriminator Bird on real data
        lossRDB = criterion(out, bird_label)
        lossRDB.backward()

        #train discriminator Bird on fake data
        noise = torch.randn(batch_size, 128, 1, 1).to(device)
        fakeData = G(noise)
        out2 = DB(fakeData.detach())
        lossFDB = criterion(out2, fake_label)
        lossFDB.backward()
        db_optimiser.step()

        out = DH(x) #train discriminator horse on real data
        lossRDH = criterion(out, bird_label)
        lossRDH.backward()

        #train discriminator Bird on fake data
        noise = torch.randn(batch_size, 128, 1, 1).to(device)
        fakeData = G(noise)
        out2 = DH(fakeData.detach())
        lossFDH = criterion(out2, fake_label)
        lossFDH.backward()
        dh_optimiser.step()

        #collect discriminator stats
        lossDB = lossFDB.cpu().item() + lossRDB.cpu().item()
        lossDH = lossFDH.cpu().item() + lossRDH.cpu().item()
        loss_arr_db = np.append(loss_arr_db, lossDB)
        loss_arr_dh = np.append(loss_arr_dh, lossDH)

        #train generator on fake data bird
        out3 = DB(fakeData) #redo after the optimizer step
        lossFGB = criterion(out3, real_label)
        lossFGB.backward(retain_graph=True)
        g_optimiser.step()

        g_optimiser.zero_grad()
        #train generator on fake data horse
        out3 = DH(fakeData) #redo after the optimizer step
        lossFGH = criterion(out3, real_label)
        lossFGH.backward()
        g_optimiser.step()

        #collect generator stats
        loss_arr_g = np.append(loss_arr_g, lossFGH.cpu().item() + lossFGB.cpu().item())

    vis.line(X=[epoch], Y=[loss_arr_g.mean().item()], win="GLoss", opts=dict(title="Generator Loss"), update="append")
    vis.line(X=[epoch], Y=[loss_arr_db.mean().item()], win="DBLoss", opts=dict(title="Discriminator Loss"), update="append")
    vis.line(X=[epoch], Y=[loss_arr_dh.mean().item()], win="DHLoss", opts=dict(title="Discriminator Loss"), update="append")

    #eval print 5x5 grid
    noise = torch.randn(batch_size, 128, 1, 1).to(device)
    fakeData = G(noise)
    t = torchvision.utils.make_grid(fakeData)
    vis.image(t, win="image", opts=dict(title="image"))

    epoch += 1
    torch.save({'G':G.state_dict(), 
        'G_optimiser': g_optimiser.state_dict(), 
        'epoch':epoch,
        'DB':DB.state_dict(), 
        'DB_optimiser': db_optimiser.state_dict(),
        'DH':DH.state_dict(), 
        'DH_optimiser': dh_optimiser.state_dict(),}, fileName)


