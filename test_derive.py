import torch
from torch import nn

from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pdb


class net(torch.nn.Module):
    def __init__(self, widths, poly_order, model_order):
        super().__init__()
        self.model_order = model_order
        self.poly_order = poly_order
        self.widths = widths
        self.we = []
        self.ae = []
        self.wd = []
        self.ad = []
        self.E = []
        self.theta = []

    def build_encoder(self):
        for i in range(len(self.widths[:-1])):
            self.we.append(nn.Linear(self.widths[i], self.widths[i+1]))
            nn.init.xavier_uniform_(self.we[i].weight, gain=nn.init.calculate_gain('relu'))
            self.ae.append(nn.ReLU())
        return

    def build_decoder(self):
        for i in range(len(self.widths[:-1])):
            self.wd.append(nn.Linear(self.widths[-i-1], self.widths[-i-2]))
            nn.init.xavier_uniform_(self.we[i].weight, gain=nn.init.calculate_gain('relu'))
            self.ad.append(nn.ReLU())
        return

    def encoder(self, x):
        for i, w in enumerate(self.we):
            x = w(x)
            x = self.ae[i](x)
        return x

    def decoder(self, x):
        for i, w in enumerate(self.wd):
            x = w(x)
            x = self.ad[i](x)
        return x

    def build_sindy(self, z, output_dim):
        self.theta = sindy_library(z, self.poly_order)
        self.E = nn.Linear(len(self.theta), output_dim)
        return self.theta, self.E

    def forward(self, x, dx, ddx):
        # W = torch.rand(4, 10)
        # z = torch.sigmoid(torch.matmul(x, W))

        z = self.encoder(x)

        if model_order == 1:
            dz = self.calc_dz(x, dx, 1)
            theta, E = self.build_sindy(z, z.shape[0])
            dzb = E(theta)

            yb = self.decoder(z)
            dyb = self.calc_dz(z, dzb, 0)

            return z, dz, dzb, yb, dyb

        else:
            dz, ddz = self.calc_ddz(x, dx, ddx, 1)
            theta, E = self.build_sindy(torch.cat((z, dz)), z.shape[0])
            ddzb = E(theta)

            yb = self.decoder(z)

            pdb.set_trace()

            dyb, ddyb = self.calc_ddz(z, dz, ddzb, 0)

            return z, dz, ddzb, yb, dyb, ddyb

    def calc_gx_enc(self, x):
        gx = torch.autograd.functional.jacobian(self.encoder, x, create_graph=True)
        return gx

    def calc_ggx_enc(self, x):
        ggx = torch.autograd.functional.jacobian(self.calc_gx_enc, x)
        return torch.diagonal(ggx, dim1=1, dim2=2)

    def calc_gx_dec(self, x):
        gx = torch.autograd.functional.jacobian(self.decoder, x, create_graph=True)
        return gx

    def calc_ggx_dec(self, x):
        ggx = torch.autograd.functional.jacobian(self.calc_gx_dec, x)
        return torch.diagonal(ggx, dim1=1, dim2=2)

    def calc_dz(self, x, dx, is_encoder):
        if is_encoder:
            gx = self.calc_gx_enc(x)
        else:
            gx = self.calc_gx_dec(x)
        return torch.matmul(gx, dx)

    def calc_ddz(self, x, dx, ddx, is_encoder):
        if is_encoder:
            gx = self.calc_gx_enc(x)
            ggx = self.calc_ggx_enc(x)
        else:
            gx = self.calc_gx_dec(x)
            ggx = self.calc_ggx_dec(x)

        dz = torch.matmul(gx, dx)
        ddz = torch.matmul(gx, ddx) + torch.matmul(ggx, dx)

        return dz, ddz


def sindy_library(z, poly_order):

    latent_dim = z.shape[0]

    library = []

    for i in range(latent_dim):
        library.append(torch.ones(1))

    for i in range(latent_dim):
        library.append(z[i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(z[i] * z[j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[i] * z[j] * z[k])

    return torch.FloatTensor(library)


W = torch.rand(4, 10)
x = torch.rand(64)
dx = torch.rand(64)
ddx = torch.rand(64)

widths = [64, 32, 16, 8]
poly_order = 3
model_order = 1

mynet = net(widths, poly_order, model_order)
mynet.build_encoder()
mynet.build_decoder()

z = mynet(x, dx, ddx)

pdb.set_trace()
# gx = mynet.calc_gx_enc(x)
# ggx = mynet.calc_ggx_enc(x)
# dz = mynet.calc_dz(x, dx, 1)
# ddz = mynet.calc_ddz(x, dx, ddx, 1)

# library = sindy_library(z, 1)
#
# mynet.build_sindy(z)
