import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pdb
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from network import Net, sindy_library
from load_data import get_data

# Load in the data
x, dx = get_data()

# Network Parameters
N, input_dim = x.shape
latent_dim = 8
widths = [input_dim, 32, 16, 8]
poly_order = 3
model_order = 1
temp = sindy_library(torch.ones(latent_dim), poly_order)
sindy_dim = temp.shape[0]

net = Net(widths, poly_order, model_order, input_dim, latent_dim, sindy_dim)
net.build_encoder()
net.build_decoder()
net.build_sindy()

epochs = 100
losses = []
loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(),
                             lr = 1e-5,
                             weight_decay = 1e-8)

# Train
for epoch in range(epochs):
    loop = tqdm(range(N), total=N, leave=False)
    for i in loop:
        z, dz, dzb, xb, dxb = net(x[i, :], dx[i, :], 0)

        loss = loss_function(xb, x[i, :])
        loss += loss_function(dz, dzb)
        loss += loss_function(dx[i, :], dxb)
        loss += 1e-2*torch.linalg.norm(net.E.weight, ord=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mask = net.E.weight > 0.1
            net.E.weight *= mask

        # Update progress bar
        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss = loss.item())

    losses.append(loss.detach().numpy())

plt.plot(losses)
plt.show()

pdb.set_trace()