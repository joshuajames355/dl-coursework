import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from models import VAE

params = torch.load('save.chkpt')
batch_size  = 64
n_channels  = 3
latent_size = 512

A = VAE(n_channels, latent_size)
optimiser = torch.optim.Adam(A.parameters(), lr=0.001)

A.load_state_dict(params['A'])
optimiser.load_state_dict(params['optimiser'])
epoch = params['epoch']

z = torch.randn(batch_size, latent_size,1,1)
print(z.size())
g = A.decode(z)
print(g.size())

# plot some examples
plt.rcParams['figure.dpi'] = 100
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(g[:8]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
plt.show()
plt.pause(0.0001)