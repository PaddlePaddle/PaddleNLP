import numpy as np
import random
import paddle
# import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
