from config import opt
import numpy as np
import torch
import random
import os

from solver import Solver

def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True

def main(args):
    net = Solver(args)
    net.train()


if __name__ == "__main__":
    setup_seed(opt.seed)
    main(opt)
