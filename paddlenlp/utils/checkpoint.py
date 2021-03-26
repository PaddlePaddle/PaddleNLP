# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import random
import numpy as np
import paddle
import paddlenlp
from paddlenlp.utils.log import logger

__all__ = ['save_checkpoint', 'load_checkpoint']


def print_arguments(args, checkpoint_time, checkpoint_step):
    """print arguments"""
    print('-------- Checkpoint time:{}, step:{} ---------'.format(
        checkpoint_time, checkpoint_step))
    print('----------- Checkpoint Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('--------------------------------------------')


def save_checkpoint(checkpoint_path,
                    checkpoint_step,
                    optimizer,
                    lr_scheduler=None,
                    args=None):
    """
    Save the model checkpoint to the state_dict, and the model checkpoint include those message
    1) optimizer state_dict 
    2) lr_scheduler state_dict
    3) random state
    4) the arguments when training 
    5) the time when saving the checkpoint
    Args:
        checkpoint_path (object: `str`):
            The save path for the checkpoint.
        checkpoint_step(object: `int`):
            The step num for the checkpoint, just used by saving the checkpoint.
        optimizer (object: `paddle.optimizer.Optimizer`):
            The optimizer used by in the process of training.
        lr_scheduler (object: `paddle.optimizer.lr.LRScheduler`, optional, default to None):
            The dynamic learning_rate scheduler in the process of training. 
            If do not use the `LRScheduler`, just set the value to None.
        args (object: , optional, default to None):
            The training argument used in the process of training. 
    """
    worker_index = paddle.distributed.get_rank()
    if worker_index == 0:
        state_dict = {}
        # First step. save the optimizer state_dict        
        optimizer_state_dict = optimizer.state_dict()

        # Second step, save the lr_scheduler state_dict
        if lr_scheduler is not None and isinstance(
                lr_scheduler, paddle.optimizer.lr.LRScheduler):
            state_dict["lr_scheduler_state"] = lr_scheduler.state_dict()

        # Third step, save the rondom seed 
        state_dict['random_rng_state'] = random.getstate()
        state_dict['np_rng_state'] = np.random.get_state()
        # TODO(wawltor) Save the cuda seed for generator if we need 

        # Fourth step, save the training arguments
        if args is not None:
            state_dict['args'] = args
        checkpoint_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        state_dict['checkpoint_time'] = checkpoint_time
        state_dict['checkpoint_step'] = checkpoint_step

        # Last setp , save the state_dict to output path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        paddle.save(state_dict,
                    os.path.join(checkpoint_path, "model_state.pdrand_lr"))
        paddle.save(optimizer_state_dict,
                    os.path.join(checkpoint_path, "model_state.pdopt"))

        logger.info("Save checkpoint at step-{} done.".format(checkpoint_step))


def load_checkpoint(checkpoint_path,
                    checkpoint_step,
                    optimizer,
                    lr_scheduler=None):
    """
    Load the checkpoint from the save path, add will load state_dict and print the messsage
    1) the time when saving the checkpoint
    2) the arguments when saving the checkpoint
    3) optimizer state_dict
    4) lr_scheduler state_dict
    5) random state
    Args:
        checkpoint_path (object: `str`):
            The load path for the checkpoint.
        checkpoint_step(object: `int`):
            The step num for the checkpoint, will return the step from the state_dict.
        optimizer (object: `paddle.optimizer.Optimizer`):
            The optimizer used by in the process of training.
        lr_scheduler (object: `paddle.optimizer.lr.LRScheduler`, optional, default to None):
            The dynamic learning_rate scheduler in the process of training. 
            If do not want to restore `LRScheduler`, just set the value to None.
    """
    opt_load_path = os.path.join(checkpoint_path, "model_state.pdopt")
    rand_lr_load_path = os.path.join(checkpoint_path, "model_state.pdrand_lr")
    # First step, check the checkpoint path exists
    if not os.path.exists(opt_load_path) or not os.path.exists(
            rand_lr_load_path):
        logger.info(
            "The optimizer and random state file is not in {} directory, just skip load the those state.".
            format(checkpoint_path))
        return checkpoint_step

    optimizer_state_dict = paddle.load(opt_load_path)
    state_dict = paddle.load(rand_lr_load_path)

    # Second step, print the argument when saving the checkpoint
    if 'args' in state_dict.keys():
        args = state_dict['args']
        checkpoint_time = state_dict['checkpoint_time']
        checkpoint_step = state_dict['checkpoint_step']
        print_arguments(args, checkpoint_time, checkpoint_step)

    # Third step, load the optimizer state_dict 
    optimizer.set_state_dict(optimizer_state_dict)

    # Fourth step, load the lr_scheduler state_dict 
    if lr_scheduler is not None and isinstance(lr_scheduler,
                                               paddle.optimizer.lr.LRScheduler):
        lr_scheduler.set_state_dict(state_dict['lr_scheduler_state'])

    # Fifth step, load the random state 
    random.setstate(state_dict['random_rng_state'])
    np.random.set_state(state_dict['np_rng_state'])

    # Last step, load the global step 
    checkpoint_step = state_dict['checkpoint_step']

    logger.info("Load the checkpoint from the step-{} done.".format(
        checkpoint_step))
    return checkpoint_step
