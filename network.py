import torch
from torch import nn
import numpy as np
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pdb


class Net(torch.nn.Module):
    def __init__(self, widths, poly_order, model_order, input_dim, latent_dim, sindy_dim):
        super().__init__()
        self.model_order = model_order
        self.poly_order = poly_order
        self.widths = widths
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sindy_dim = sindy_dim
        self.we = []
        self.ae = []
        self.wd = []
        self.ad = []
        self.E = []

    def build_encoder(self):
        for i in range(len(self.widths[:-1])):
            self.we.append(nn.Linear(self.widths[i], self.widths[i+1]))
            nn.init.xavier_uniform_(self.we[i].weight, gain=nn.init.calculate_gain('sigmoid'))
            if i < len(self.widths[:-1]) - 1:
                self.ae.append(nn.Sigmoid())
            else:
                self.ae.append(nn.Identity())
        return

    def build_decoder(self):
        for i in range(len(self.widths[:-1])):
            self.wd.append(nn.Linear(self.widths[-i-1], self.widths[-i-2]))
            nn.init.xavier_uniform_(self.we[i].weight, gain=nn.init.calculate_gain('sigmoid'))
            if i < len(self.widths[:-1]) - 1:
                self.ad.append(nn.Sigmoid())
            else:
                self.ad.append(nn.Identity())
        return

    def build_sindy(self):
        self.E = nn.Linear(self.sindy_dim, self.latent_dim)
        nn.init.ones_(self.E.weight)
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

    def forward(self, x, dx, ddx):
        # W = torch.rand(4, 10)
        # z = torch.sigmoid(torch.matmul(x, W))

        z = self.encoder(x)

        if self.model_order == 1:
            dz = self.calc_dz(x, dx, 1)
            theta = sindy_library(z, self.poly_order)
            dzb = self.E(theta)

            xb = self.decoder(z)
            dxb = self.calc_dz(z, dzb, 0)

            return z, dz, dzb, xb, dxb

        else:
            dz, ddz = self.calc_ddz(x, dx, ddx, 1)
            theta = sindy_library(torch.cat((z, dz)), self.poly_order)
            ddzb = self.E(theta)

            xb = self.decoder(z)

            pdb.set_trace()

            dxb, ddxb = self.calc_ddz(z, dz, ddzb, 0)

            return z, dz, ddzb, xb, dxb, ddxb

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

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[i] * z[j] * z[k] * z[p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(z[i]*z[j]*z[k]*z[p]*z[q])

    return torch.FloatTensor(library)


# t = torch.arange(0, 3, 0.01)
#
# x = torch.sin(t)
# x = torch.stack((x, x))
#
# dx = torch.cos(t)
# dx = torch.stack((dx, dx))
#
# # W = torch.rand(4, 10)
# # x = torch.rand(64)
# # dx = torch.rand(64)
# # ddx = torch.rand(64)
#
# input_dim = 2
# latent_dim = 8
# widths = [2, 32, 16, 8]
# poly_order = 3
# model_order = 1
#
# temp = sindy_library(torch.ones(latent_dim), poly_order)
# sindy_dim = temp.shape[0]
#
# # Build network
#
# mynet = net(widths, poly_order, model_order, input_dim, latent_dim, sindy_dim)
# mynet.build_encoder()
# mynet.build_decoder()
# mynet.build_sindy()
#
# epochs = 100
# losses = []
#
# loss_function = nn.MSELoss()
#
# optimizer = torch.optim.Adam(mynet.parameters(),
#                              lr = 1e-5,
#                              weight_decay = 1e-8)
#
# for e in range(epochs):
#     for i in range(len(t)):
#
#         print(i)
#
#         z, dz, dzb, xb, dxb = mynet(x[:, i], dx[:, i], 0)
#
#         loss = loss_function(xb, x[:, i])
#         loss += loss_function(dz, dzb)
#         loss += loss_function(dx[:, i], dxb)
#         loss += torch.linalg.norm(mynet.E.weight, ord=1)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         with torch.no_grad():
#             mask = mynet.E.weight > 0.1
#             mynet.E.weight *= mask
#
#     losses.append(loss.detach().numpy())
#
# plt.plot(losses)
# plt.show()
#
#
#
# # gx = mynet.calc_gx_enc(x)
# # ggx = mynet.calc_ggx_enc(x)
# # dz = mynet.calc_dz(x, dx, 1)
# # ddz = mynet.calc_ddz(x, dx, ddx, 1)
#
# # library = sindy_library(z, 1)
# #
# # mynet.build_sindy(z)
