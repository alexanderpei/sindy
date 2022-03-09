import torch
from torch import nn
import numpy as np
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pdb


class Net(torch.nn.Module):
    def __init__(self, widths, poly_order, model_order, input_dim, latent_dim, sindy_dim, N):
        super().__init__()
        self.model_order = model_order
        self.poly_order = poly_order
        self.widths = widths
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sindy_dim = sindy_dim
        self.encoder = nn.Sequential
        self.decoder = nn.Sequential
        self.E = []
        self.N = N

        # Sindy Layer
        self.E = nn.Linear(self.sindy_dim, self.latent_dim)
        nn.init.ones_(self.E.weight)

        # Build encoder/decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(self.widths[:-1])):

            # Encoder
            temp = nn.Linear(self.widths[i], self.widths[i+1])
            nn.init.xavier_uniform_(temp.weight, gain=nn.init.calculate_gain('sigmoid'))
            self.encoder.append(temp)

            # Decoder
            temp = nn.Linear(self.widths[-i-1], self.widths[-i-2])
            nn.init.xavier_uniform_(temp.weight, gain=nn.init.calculate_gain('sigmoid'))
            self.decoder.append(temp)

            if i < len(self.widths[:-1]) - 1:
                self.encoder.append(nn.Sigmoid())
                self.decoder.append(nn.Sigmoid())
            else:
                self.encoder.append(nn.Identity())
                self.decoder.append(nn.Identity())

    def encoder_forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x

    def decoder_forward(self, x):
        for layer in self.decoder:
            x = layer(x)

        return x

    def forward(self, x, dx, ddx):

        z = self.encoder_forward(x)

        if self.model_order == 1:
            dz = self.calc_dz(x, dx, 1)
            theta = sindy_library(z, self.poly_order, self.latent_dim, self.N)
            dzb = self.E(theta)

            xb = self.decoder_forward(z)
            dxb = self.calc_dz(z, dzb, 0)

            return z, dz, dzb, xb, dxb

        else:
            dz, ddz = self.calc_ddz(x, dx, ddx, 1)
            theta = sindy_library(torch.cat((z, dz)), self.poly_order, self.latent_dim, self.N)
            ddzb = self.E(theta)

            xb = self.decoder(z)

            dxb, ddxb = self.calc_ddz(z, dz, ddzb, 0)

            return z, dz, ddzb, xb, dxb, ddxb

    def calc_gx_enc(self, x):
        gx = torch.autograd.functional.jacobian(self.encoder_forward, x, create_graph=True)

        # Gradient within sample
        gx_ = torch.diagonal(gx, dim1=0, dim2=2)
        # Average across samples
        gx_ = torch.mean(gx_, dim=2)

        return gx_

    def calc_ggx_enc(self, x):
        ggx = torch.autograd.functional.jacobian(self.calc_gx_enc, x)
        ggx_ = torch.diagonal(ggx, dim1=1, dim2=3)
        return torch.mean(ggx_, dim=1)

    def calc_gx_dec(self, x):
        gx = torch.autograd.functional.jacobian(self.decoder_forward, x, create_graph=True)
        # Gradient within sample
        gx_ = torch.diagonal(gx, dim1=0, dim2=2)
        # Average across samples
        gx_ = torch.mean(gx_, dim=2)

        return gx_

    def calc_ggx_dec(self, x):
        ggx = torch.autograd.functional.jacobian(self.calc_gx_dec, x)
        ggx_ = torch.diagonal(ggx, dim1=1, dim2=3)
        return torch.mean(ggx_, dim=1)

    def calc_dz(self, x, dx, is_encoder):
        if is_encoder:
            gx = self.calc_gx_enc(x)
        else:
            gx = self.calc_gx_dec(x)

        return torch.matmul(dx, gx.T)

    def calc_ddz(self, x, dx, ddx, is_encoder):
        if is_encoder:
            gx = self.calc_gx_enc(x)
            ggx = self.calc_ggx_enc(x)
        else:
            gx = self.calc_gx_dec(x)
            ggx = self.calc_ggx_dec(x)

        dz = torch.matmul(dx, gx.T)
        ddz = torch.matmul(ddx, gx.T) + torch.matmul(dx, ggx.T)

        return dz, ddz


def sindy_library(z, poly_order, latent_dim, N):

    library = []

    # Constant
    for i in range(latent_dim):
        library.append(torch.ones(N, device=z.device))

    # 1st order
    if poly_order > 0:
        for i in range(latent_dim):
            library.append(z[:, i])

    # 2nd order
    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(z[:, i]*z[:,j])

    # #rd order
    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i]*z[:, j]*z[:, k])

    return torch.stack(library, axis=1)


