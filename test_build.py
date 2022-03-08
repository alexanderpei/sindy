import torch
from torchvision import datasets
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import pdb


class net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = [64, 32, 16, 8]
        self.w = [];
        self.a = [];

        for i in range(len(self.weights[:-1])):
            self.w.append(nn.Linear(self.weights[i], self.weights[i+1]))
            nn.init.xavier_uniform_(self.w[i].weight, gain=nn.init.calculate_gain('relu'))

            self.a.append(nn.ReLU())

    def forward(self, x):

        for i in range(len(self.weights[:-1])):
            x = self.w[i](x)
            x = self.a[i](x)

        return x


x = torch.rand((1, 64))

pdb.set_trace()

mynet = net()
temp = mynet(x)

pdb.set_trace()