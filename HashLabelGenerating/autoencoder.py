import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class ConvAutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=0),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(output_dim, input_dim, kernel_size=kernel_size, stride=stride, padding=0),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x, is_reconstruct):
        x = x.detach()
        y = self.encoder(x)

        if self.training:
            if is_reconstruct:
                x_reconstruct = self.decoder(y)
                loss_convlayer = self.criterion(x_reconstruct, Variable(x, requires_grad=False))
                return y.detach(), loss_convlayer
            else:
                return y.detach()
        else:
            return y.detach()

    def reconstruct(self, x):
        return self.decoder(x)


class LinearAutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.BatchNorm1d(output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim, bias=True),
            nn.BatchNorm1d(input_dim),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x, is_reconstruct):
        x = x.detach()
        y = self.encoder(x)

        if self.training:
            if is_reconstruct:
                x_reconstruct = self.decoder(y)
                loss_linearlayer = self.criterion(x_reconstruct, Variable(x, requires_grad=False))
                return y.detach(), loss_linearlayer
            else:
                return y.detach()
        else:
            return y.detach()

    def reconstruct(self, x):
        return self.decoder(x)


class LinearAutoEncoderLast(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearAutoEncoderLast, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.BatchNorm1d(output_dim),
            Activation_op(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim, bias=True),
            nn.BatchNorm1d(input_dim),
            Activation_op(),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x, is_reconstruct):
        x = x.detach()
        y = self.encoder(x)

        if self.training:
            if is_reconstruct:
                x_reconstruct = self.decoder(y)
                loss_linearlayer = self.criterion(x_reconstruct, Variable(x, requires_grad=False))
                return y.detach(), loss_linearlayer
            else:
                return y.detach()
        else:
            return y.detach()

    def reconstruct(self, x):
        return self.decoder(x)


class StackedAutoEncoder(nn.Module):

    def __init__(self, hash_len):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = ConvAutoEncoder(10, 32, 2, 2)
        self.ae2 = ConvAutoEncoder(32, 64, 2, 2)
        self.ae3 = ConvAutoEncoder(64, 128, 5, 4)
        self.ae4 = ConvAutoEncoder(128, 256, 2, 2)
        self.ae5 = LinearAutoEncoder(256 * 3 * 3, 1024)
        self.ae6 = LinearAutoEncoder(1024, 512)
        self.ae7 = LinearAutoEncoder(512, 256)
        self.ae8 = LinearAutoEncoder(256, 128)
        self.ae9 = LinearAutoEncoder(128, 96)
        self.ae10 = LinearAutoEncoderLast(96, hash_len)

    def forward(self, x, batch_size):
        if self.training:
            a1 = self.ae1(x, False)
            a2, loss_a2 = self.ae2(a1, True)
            a3 = self.ae3(a2, False)
            a4, loss_a4 = self.ae4(a3, True)
            a4 = a4.view(-1, 256 * 3 * 3)
            a5 = self.ae5(a4, False)
            a6, loss_a6 = self.ae6(a5, True)
            a7 = self.ae7(a6, False)
            a8, loss_a8 = self.ae8(a7, True)
            a9 = self.ae9(a8, False)
            a10, loss_a10 = self.ae10(a9, True)
            loss_layer_sum = 0.52*loss_a2+0.26*loss_a4+0.13*loss_a6+0.06*loss_a8+0.03*loss_a10
            return a10, self.reconstruct(a10, batch_size), loss_layer_sum

        else:
            a1 = self.ae1(x, False)
            a2 = self.ae2(a1, False)
            a3 = self.ae3(a2, False)
            a4 = self.ae4(a3, False)
            a4 = a4.view(-1, 256 * 3 * 3)
            a5 = self.ae5(a4, False)
            a6 = self.ae6(a5, False)
            a7 = self.ae7(a6, False)
            a8 = self.ae8(a7, False)
            a9 = self.ae9(a8, False)
            a10 = self.ae10(a9, False)
            return a10

    def reconstruct(self, x, batch_size):
        a9_reconstruct = self.ae10.reconstruct(x)
        a8_reconstruct = self.ae9.reconstruct(a9_reconstruct)
        a7_reconstruct = self.ae8.reconstruct(a8_reconstruct)
        a6_reconstruct = self.ae7.reconstruct(a7_reconstruct)
        a5_reconstruct = self.ae6.reconstruct(a6_reconstruct)
        a4_reconstruct = self.ae5.reconstruct(a5_reconstruct)
        a4_reconstruct = a4_reconstruct.reshape(batch_size, 256, 3, 3)
        a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
        a2_reconstruct = self.ae3.reconstruct(a3_reconstruct)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct

class Activation_op(nn.Module):
    def __init__(self):
        super(Activation_op, self).__init__()

    def forward(self, x):
        g = 0.2
        x = torch.tanh(g * x)
        return x
