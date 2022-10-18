import torch
import torch.nn as nn


class OBEBlock(nn.Module):

    def __init__(self, img_size):
        super(OBEBlock, self).__init__()

        self.img_size = img_size
        # P parameters for all image - defined as trainable parameters
        self.P = nn.Parameter(torch.rand(self.img_size, self.img_size))
        nn.init.orthogonal_(self.P)
        self.conv = nn.Conv2d(3, 1, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.matmul(torch.matmul(self.P, x), self.P.T)
        out = self.conv(out).view(B, -1)
        return out
