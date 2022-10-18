import os
import torch
import argparse
import numpy as np

class DataGather(object):
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg: [] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def noise_sample(n_con_c, n_z, batch_size, device):
    noise = torch.randn(batch_size, n_z, device=device)

    # Random uniform between -1 and 1.
    con_c = torch.rand(batch_size, n_con_c, device=device) * 2 - 1
    noise = torch.cat((noise, con_c), dim=1)

    return noise


def make_opt(n):
    opts = []
    for i in range(n):
        opt = {
            "title": 'change c{}'.format(i + 1),
            "xlabel": 'c{}'.format(i + 1),
            "ylabel": "c'",
            "legend":  ["c'" + str(x) for x in list(range(1, n + 1))]
        }
        opts.append(opt)
    return opts

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
 
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll