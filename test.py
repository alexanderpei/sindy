import numpy as np
import scipy.io
import pdb
import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
# import pysindy as ps

# Torch use the GPU
print(torch.cuda.is_available())
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# Load in the data
data = scipy.io.loadmat('data.mat')
t = np.squeeze(data['t'])
fs = data['fs']
data = data['data']

nChan, nSub, nTime, nCond = data.shape

arr = data[:, :, :, 0]
arr = np.reshape(arr, (nChan, nSub*nTime)).T
arr = torch.Tensor(arr)
dataset = TensorDataset(arr)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# data_ = data_.to(device)

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


model = AE()
# model.to(device)
loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-5,
                             weight_decay = 1e-8)

epochs = 20
losses = []

for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for iBatch, x in loop:

        encoded, reconstructed = model(x[0])

        loss = loss_function(reconstructed, x[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().numpy())

        # Update progress bar
        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss = loss.item())


# Forward pass with a single subject's data
arr = data[:, 4, :, 0]
arr = torch.Tensor(arr)
outputs = np.zeros((nChan, nTime))
outputs_encoded = np.zeros((4, nTime))
for i in range(len(t)):
    encoded, reconstructed = model(arr[:, i])

    outputs[:, i] = reconstructed.detach().numpy()
    outputs_encoded[:, i] = encoded.detach().numpy()

fig, axs = plt.subplots(2, 2)
fig.suptitle('Reconstructed vs. Original vs. Encoded')
axs[0, 0].plot(outputs.T)
axs[0, 1].plot(arr.T)
axs[1, 0].plot(outputs_encoded.T)
axs[1, 1].plot(losses)

fig.show()

pdb.set_trace()


fig1, axs = plt.subplots(5, 6)
for iSub, ax in enumerate(axs.flat):

    print(iSub)

    if iSub <= nSub-1:
        arr = data[:, iSub, :, 0]
        arr = torch.Tensor(arr)
        outputs = np.zeros((nChan, nTime))
        outputs_encoded = np.zeros((4, nTime))
        for i in range(len(t)):
            encoded, reconstructed = model(arr[:, i])

            outputs[:, i] = reconstructed.detach().numpy()
            outputs_encoded[:, i] = encoded.detach().numpy()

        ax.plot(outputs_encoded.T)

fig1.show()


fig1, axs = plt.subplots(5, 6)
for iSub, ax in enumerate(axs.flat):

    print(iSub)

    if iSub <= nSub-1:
        arr = data[:, iSub, :, 0]
        arr = torch.Tensor(arr)
        outputs = np.zeros((nChan, nTime))
        outputs_encoded = np.zeros((4, nTime))
        for i in range(len(t)):
            encoded, reconstructed = model(arr[:, i])

            outputs[:, i] = reconstructed.detach().numpy()
            outputs_encoded[:, i] = encoded.detach().numpy()

        ax.plot(outputs.T)

fig1.show()



pdb.set_trace()