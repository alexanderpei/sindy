import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pdb
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from network_batch import Net, sindy_library
from load_data import get_data_batch

# Torch use the GPU
print(torch.cuda.is_available())
global dev
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
use_gpu = 0

# Load in the data
batch_size = 1
x, dx = get_data_batch(batch_size, use_gpu)

x = x[0:1000, :]
dx = dx[0:1000, :]

if use_gpu:
    x = x.to(device)
    dx = dx.to(device)

dataset = TensorDataset(x, dx)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

temp = next(iter(dataloader))

# Network Parameters
N, input_dim = temp[0].shape
latent_dim = 8
widths = [input_dim, 32, 16, latent_dim]
poly_order = 3
model_order = 1
temp = sindy_library(torch.ones((N, latent_dim), device=dev), poly_order, latent_dim, N)
sindy_dim = temp.shape[1]

net = Net(widths, poly_order, model_order, input_dim, latent_dim, sindy_dim, N)
if use_gpu:
    net = net.to(device)

epochs = 50
losses = []
loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(),
                             lr = 1e-3,
                             weight_decay = 1e-8)

# Train
for epoch in range(epochs):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for i, data in loop:

        x = data[0]
        dx = data[0]

        z, dz, dzb, xb, dxb = net(x, dx, 0)

        loss = loss_function(xb, x)
        loss += 0.01*loss_function(dz, dzb)
        loss += 0.5*loss_function(dx, dxb)
        loss += torch.sum(torch.abs(net.E.weight))/torch.numel(net.E.weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mask = net.E.weight > 0.1
            net.E.weight *= mask

        # Update progress bar

        n_weights = torch.sum(mask).detach().numpy()
        loss_ = np.round(loss.item(), 2)

        temp = net.E.weight.detach().numpy().flatten()
        idx = np.where(temp > 0.1)
        min_ = np.round(np.min(temp[idx]), 2)

        # min_ = np.round(torch.min(net.E.weight).detach().numpy(), 2)
        max_ = np.round(torch.max(net.E.weight).detach().numpy(), 2)

        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss = loss_, n_weights = n_weights, min=min_, max=max_)

    losses.append(loss.detach().numpy())

plt.plot(losses)
plt.show()

pdb.set_trace()