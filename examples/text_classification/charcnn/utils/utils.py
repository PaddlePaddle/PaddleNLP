import numpy as np
import random
import paddle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)