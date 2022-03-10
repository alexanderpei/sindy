import torch
import scipy.io
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pysindy as ps
import pdb
import matplotlib.pyplot as plt


def get_data():
    # Load in the data
    data = scipy.io.loadmat('data.mat')
    t = np.squeeze(data['t'])
    fs = data['fs']
    data = data['data']

    # dx = ps.SmoothedFiniteDifference()._differentiate(data[0, 0, :, 0], t)
    #
    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(t, data[0, 0, :, 0])
    # axs[1].plot(t, dx)
    # fig.show()

    nChan, nSub, nTime, nCond = data.shape

    x = data[:, 3, :, 0]
    nSub = 1

    # Calculate dx
    dx = np.zeros(x.shape)

    for iChan in range(nChan):
        for iSub in range(nSub):
            dx[iChan, :] = ps.SmoothedFiniteDifference()._differentiate(x[iChan, :], t)

    x = np.reshape(x, (nChan, nSub*nTime)).T
    dx = np.reshape(dx, (nChan, nSub*nTime)).T

    x = torch.Tensor(x)
    dx = torch.Tensor(dx)

    return x, dx


def get_data_batch(batch_size, use_gpu):

    # Load in the data
    data = scipy.io.loadmat('data.mat')
    t = np.squeeze(data['t'])
    fs = data['fs']
    data = data['data']

    nChan, nSub, nTime, nCond = data.shape

    x = data[:, :, :, 0]

    # Calculate dx
    dx = np.zeros(x.shape)

    for iChan in range(nChan):
        for iSub in range(nSub):
            dx[iChan, iSub, :] = ps.SmoothedFiniteDifference()._differentiate(x[iChan, iSub, :], t)

    x = np.reshape(x, (nChan, nSub*nTime)).T
    dx = np.reshape(dx, (nChan, nSub*nTime)).T

    x = torch.Tensor(x)
    dx = torch.Tensor(dx)

    return x, dx, t
