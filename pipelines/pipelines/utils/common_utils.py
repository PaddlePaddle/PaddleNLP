from typing import Any, Iterator, Tuple, List

import logging
import os
import pickle
import random
import signal
from copy import deepcopy
from itertools import islice
import numpy as np
import paddle

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int, deterministic_cudnn: bool = False) -> None:
    """
    Setting multiple seeds to make runs reproducible.

    Important: Enabling `deterministic_cudnn` gives you full reproducibility with CUDA,

    :param seed:number to use as seed
    :param deterministic_paddle: Enable for full reproducibility when using CUDA. Caution: might slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_cudnn:
        # Todo(tianxin04): set cudnn deterministic
        pass


def initialize_device_settings(use_cuda: bool,
                               local_rank: int = -1,
                               multi_gpu: bool = True) -> Tuple[List[str], int]:
    """
    Returns a list of available devices.

    :param use_cuda: Whether to make use of CUDA GPUs (if available).
    :param local_rank: Ordinal of device to be used. If -1 and multi_gpu is True, all devices will be used.
    :param multi_gpu: Whether to make use of all GPUs (if available).
    """

    if not use_cuda:
        devices = [paddle.set_device("cpu")]
        n_gpu = 0
    elif local_rank == -1:
        if 'gpu' in paddle.get_device():
            if multi_gpu:
                devices = [
                    paddle.set_device('gpu:{}'.format(device))
                    for device in range(paddle.device.cuda.device_count())
                ]
                n_gpu = paddle.device.cuda.device_count()
            else:
                devices = [paddle.set_device("gpu")]
                n_gpu = 1
        else:
            devices = [paddle.set_device("cpu")]
            n_gpu = 0
    else:
        devices = [paddle.set_device('gpu:{}'.format(local_rank))]
        n_gpu = 1

    logger.info(
        f"Using devices: {', '.join([str(device) for device in devices]).upper()}"
    )
    logger.info(f"Number of GPUs: {n_gpu}")
    return devices, n_gpu


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


def try_get(keys, dictionary):
    try:
        for key in keys:
            if key in dictionary:
                ret = dictionary[key]
                if type(ret) == list:
                    ret = ret[0]
                return ret
    except Exception as e:
        logger.warning(f"Cannot extract from dict {dictionary} with error: {e}")
    return None
