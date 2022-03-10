import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pdb
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from network_batch_jacob import Net, sindy_library, sindy_simulate
from load_data import get_data_batch
from scipy.integrate import odeint
import os
import time
import sys

ireg = sys.argv[1]
lambas = np.logspace(-2, -5, 20)

# Torch use the GPU
print(torch.cuda.is_available())
# if torch.cuda.is_available():
#     dev = "cuda:0"
# else:
#     dev = "cpu"

use_gpu = 0
if use_gpu:
    dev = 'cuda:0'
    device = torch.device(dev)
else:
    dev = 'cpu'
    device = torch.device(dev)

# Load in the data
batch_size = 32
x, dx, t = get_data_batch(batch_size, use_gpu)

with torch.no_grad():
    x_mean = torch.sum(torch.square(x))/torch.numel(x)
    dx_mean = torch.sum(torch.square(dx))/torch.numel(dx)
    ratio = x_mean.numpy() / dx_mean.numpy()

print('Ratio: {:0.6f}'.format(ratio))

x = x[0:100, :]
dx = dx[0:100, :]

if use_gpu:
    x = x.to(device)
    dx = dx.to(device)

dataset = TensorDataset(x, dx)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

temp = next(iter(dataloader))

# Network Parameters
N, input_dim = temp[0].shape
latent_dim = 8
widths = [input_dim, 128, 64, 32, 16, latent_dim]
poly_order = 2
model_order = 1
temp = sindy_library(torch.ones((N, latent_dim), device=dev), poly_order, latent_dim, device)
sindy_dim = temp.shape[1]

net = Net(widths, poly_order, model_order, input_dim, latent_dim, sindy_dim, N, device)
if use_gpu:
    net = net.to(device)

epochs = 50
losses = []
losses_unreg = []
loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(),
                             lr = 1e-3,
                             )

# Train
for epoch in range(epochs):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for i, data in loop:

        x = data[0]
        dx = data[1]

        z, dz, dzb, xb, dxb = net(x, dx)

        loss = loss_function(xb, x)
        loss += 1/20000*loss_function(dz, dzb)
        loss += 1/2000*loss_function(dx, dxb)
        loss_unreg = np.round(loss.item(), 2)
        loss += lambas[int(ireg)]*torch.norm(net.E.weight, p=1)/torch.numel(net.E.weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mask = net.E.weight > 0.1
            net.E.weight *= mask

        # Update progress bar

        n_weights = torch.sum(mask).cpu().detach().numpy()
        loss_ = np.round(loss.item(), 2)

        temp = net.E.weight.cpu().detach().numpy().flatten()
        idx = np.where(temp > 0.1)
        min_ = np.round(np.min(temp[idx]), 2)

        # min_ = np.round(torch.min(net.E.weight).detach().numpy(), 2)
        max_ = np.round(torch.max(net.E.weight).cpu().detach().numpy(), 2)

        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss=loss_, loss_unreg=loss_unreg, n_weights=n_weights, min=min_, max=max_)

    # print('Epoch: {:d}/{:d} -- Loss: {:.2f} -- Loss_unreg: {:.2f} -- n_weights: {:.2f} -- min: {:.2f} -- max: {:.2f}'.format(epoch, epochs, loss_, loss_unreg, n_weights, min_, max_), flush=True)

    losses_unreg.append(loss_unreg)
    losses.append(loss_)

torch.save(net.state_dict(), os.path.join(os.getcwd(), 'model_' + ireg))

fig, axs = plt.subplots(1, 2)
axs[0].plot(losses)
axs[1].plot(losses_unreg)
fig.show()

plt.plot(losses)
plt.show()

# Sindy simulate
z0 = np.random.rand((latent_dim))
sol = odeint(sindy_simulate, z0, t, args=(poly_order, latent_dim, net.E), full_output=True)

plt.plot(sol)
plt.show()