import torch
import numpy as np
import torch.nn as nn


class DCTBlock(nn.Module):
    def __init__(self, img_size, device):
        super(DCTBlock, self).__init__()
        self.FloatTensor = torch.cuda.FloatTensor
        self.A = self.create(img_size)
        self.A = torch.from_numpy(self.A).type(self.FloatTensor).to(device)
        self.conv = nn.Conv2d(3, 1, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.matmul(torch.matmul(self.A, x), self.A.T)
        out = self.conv(out).view(B, -1)
        return out

    def create(self, size):
        H = np.arange(0, size).reshape(-1, 1)
        W = (np.arange(0, size).reshape(1, -1) + 0.5) * np.pi / size
        A = np.cos(H * W)
        cu = np.ones((size, size))
        cu = cu * 2 / np.sqrt(size)
        cu[0, :] /= 2
        A = np.multiply(cu, A)
        return A
