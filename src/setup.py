import os
import random

import numpy as np
import torch


def set_random_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def setup(opt):
    if opt.random_seed is not None:
        set_random_seed(opt.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")
