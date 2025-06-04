# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import numpy as np
import paddle
from paddle import pir
from paddle.base import core, framework
from paddle.base.dygraph import base as imperative_base
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.base.libpaddle import DataType
from paddle.optimizer.adamw import AdamW

class ApolloAdamW(AdamW):
    """
    Apollo optimizer combines AdamW-level performance with SGD-like memory efficiency.
    
    It uses a low-rank auxiliary space for approximated channel-wise or tensor-wise gradient scaling.
    This implementation is based on the paper "APOLLO: SGD-like Memory, AdamW-level Performance"
    https://arxiv.org/pdf/2412.05270
    
    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update parameters.
            Default: 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            Default: 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            Default: 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            Default: 1e-8.
        parameters (list|tuple, optional): List/Tuple of Tensor to update to minimize loss.
            Default: None.
        weight_decay (float, optional): The weight decay coefficient, 
            it can be float or Tensor. Default: 0.01.
        apply_decay_param_fun (function|None, optional): If it is not None,
            only tensors that makes apply_decay_param_fun(name) returns
            true will be updated. Default: None.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy.
            Default: None.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. In contrast, the lazy mode
            only updates accumulators when a sparse gradient appears.
            Default: False.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating.
            Default: False.
        rank (int, optional): The rank of the auxiliary subspace.
            Default: 256 for Apollo, 1 for Apollo-Mini.
        scale_type (str, optional): The type of gradient scaling:
            'channel': Applies gradient scaling at the channel level (Apollo)
            'tensor': Applies gradient scaling at the tensor level (Apollo-Mini)
            Default: 'channel'.
        scale (int, optional): Scaling factor for gradient updates.
            Default: 1 for Apollo, 128 for Apollo-Mini.
        scale_front (bool, optional): Whether to apply norm-growth limiter before scaling.
            Default: False.
    """
    def __init__(self, 
                 learning_rate=0.001,
                 beta1=0.9, 
                 beta2=0.999, 
                 epsilon=1e-8,
                 parameters=None, 
                 weight_decay=0.01,
                 apply_decay_param_fun=None,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
                 rank=256,
                 scale_type='channel',
                 scale=1,
                 scale_front=False):
        super(ApolloAdamW, self).__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            weight_decay=weight_decay,
            apply_decay_param_fun=apply_decay_param_fun,
            grad_clip=grad_clip,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision)
        
        self.rank = rank
        self.scale_type = scale_type
        self.scale = scale
        self.scale_front = scale_front
        
    def _add_moments_pows(self, p):
        # To be implemented
        pass
    
    def _append_optimize_op(self, block, param_and_grad):
        # To be implemented
        pass
        
    def apollo_update(self, param, grad, learning_rate, moment1, beta1_pow, beta2_pow,
                    master_weight, skip_update, beta1, beta2, epsilon, 
                    lr_ratio, coeff, with_decay, multi_precision):
        # To be implemented
        pass


class ApolloMiniAdamW(ApolloAdamW):
    """
    Apollo-Mini optimizer is an extreme memory-efficient version of Apollo
    that applies tensor-wise gradient scaling using only a rank-1 auxiliary sub-space.
    
    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update parameters.
            Default: 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            Default: 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            Default: 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            Default: 1e-8.
        parameters (list|tuple, optional): List/Tuple of Tensor to update to minimize loss.
            Default: None.
        weight_decay (float, optional): The weight decay coefficient, 
            it can be float or Tensor. Default: 0.01.
        apply_decay_param_fun (function|None, optional): If it is not None,
            only tensors that makes apply_decay_param_fun(name) returns
            true will be updated. Default: None.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy.
            Default: None.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. In contrast, the lazy mode
            only updates accumulators when a sparse gradient appears.
            Default: False.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating.
            Default: False.
        scale (int, optional): Scaling factor for gradient updates.
            Default: 128.
        scale_front (bool, optional): Whether to apply norm-growth limiter before scaling.
            Default: False.
    """
    def __init__(self, 
                 learning_rate=0.001,
                 beta1=0.9, 
                 beta2=0.999, 
                 epsilon=1e-8,
                 parameters=None, 
                 weight_decay=0.01,
                 apply_decay_param_fun=None,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
                 scale=128,
                 scale_front=False):
        super(ApolloMiniAdamW, self).__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            weight_decay=weight_decay,
            apply_decay_param_fun=apply_decay_param_fun,
            grad_clip=grad_clip,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision,
            rank=1,
            scale_type='tensor',
            scale=scale,
            scale_front=scale_front)
