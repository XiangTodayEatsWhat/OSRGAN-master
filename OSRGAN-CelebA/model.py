import torch
from config import opt
import torch.nn as nn
import torch.nn.functional as F
from Blocks.DCT import DCTBlock
from Blocks.OrthogonalBasisExpansion import OBEBlock


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_x = nn.Sequential(
            nn.Conv2d(opt.z_dim + opt.code_dim, 1024, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 256, 8, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = self.conv_x(x)

        img = torch.sigmoid(x)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, 8, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1, inplace=True)

        return x


class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = self.conv(x)

        return output.squeeze(dim=-1).squeeze(dim=-1)


class QHead(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = max(1, opt.code_dim)
        self.conv1 = nn.Conv2d(1024, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv_mu = nn.Conv2d(hidden_dim, max(1, opt.code_dim), 1)
        self.conv_var = nn.Conv2d(hidden_dim, max(1, opt.code_dim), 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return mu, var


class OBEHead(nn.Module):
    def __init__(self, img_size, model_type, device):
        super().__init__()
        if model_type == 'OBE':
            self.infer = OBEBlock(img_size=img_size)
        elif model_type == 'DCT':
            self.infer = DCTBlock(img_size=img_size, device=device)

    def forward(self, x):
        coe_c = self.infer(x)

        return coe_c
