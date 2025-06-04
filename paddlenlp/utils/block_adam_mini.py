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

"""
Implementation of Block Adam-mini optimizer based on the paper:
"Adam-mini: Parameter-Efficient Fine-tuning through Optimizer State Structured Sparsity"
(https://arxiv.org/pdf/2406.16793)
"""

import numpy as np
import paddle
from paddle import pir
from paddle.base import core, framework
from paddle.base.dygraph import base as imperative_base
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.base.libpaddle import DataType
from paddle.optimizer.adamw import AdamW

class BlockAdamMini(AdamW):
    """
    Implementation of Block Adam-mini optimizer.
    
    Block Adam-mini is a parameter-efficient fine-tuning optimizer that uses structured sparsity
    in optimizer states to significantly reduce memory consumption. This implementation divides the
    parameters into blocks and maintains only one optimizer state per block, achieving memory
    reduction of approximately 50% compared to AdamW.

    Based on the paper "Adam-mini: Parameter-Efficient Fine-tuning through Optimizer State Structured Sparsity"
    (https://arxiv.org/pdf/2406.16793)
    
    Args:
        learning_rate (float|LRScheduler): The learning rate used to update parameters.
            Default: 0.001.
        beta1 (float): The exponential decay rate for the 1st moment estimates.
            Default: 0.9.
        beta2 (float): The exponential decay rate for the 2nd moment estimates.
            Default: 0.999.
        epsilon (float): A small float value for numerical stability. Default: 1e-8.
        parameters (list|tuple): List/Tuple of parameters to optimize. Default: None.
        weight_decay (float): The weight decay coefficient. Default: 0.01.
        apply_decay_param_fun (function|None): Function to specify which parameters to apply decay.
            Default: None.
        grad_clip (GradientClipBase): Gradient cliping strategy. Default: None.
        name (str): Name for the optimizer. Default: None.
        lazy_mode (bool): Whether to use lazy mode for sparse updates. Default: False.
        multi_precision (bool): Whether to use multi-precision during weight updating. Default: False.
        block_size (int): Size of each block for state aggregation. Default: 32.
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
                 name=None,
                 lazy_mode=False,
                 multi_precision=False,
                 block_size=32):
        super(BlockAdamMini, self).__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            weight_decay=weight_decay,
            apply_decay_param_fun=apply_decay_param_fun,
            grad_clip=grad_clip,
            name=name,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision)
        
        self.block_size = block_size

    def _add_moments_pows(self, p):
        """
        Add moment and power accumulators for the parameter.
        For BlockAdamMini, we create sparse moment accumulators based on blocks.
        
        Args:
            p (Variable): The parameter to add accumulators for.
        """
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            acc_dtype = DataType.FLOAT32 if in_pir_mode() else paddle.float32

        # Create first moment accumulator
        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        
        # Create second moment accumulator with reduced shape (one per block)
        orig_shape = list(p.shape)
        if len(orig_shape) >= 1:
            # Calculate number of blocks
            total_elements = np.prod(orig_shape)
            num_blocks = max(1, int(total_elements / self.block_size))
            block_shape = [num_blocks]
            
            # Add the reduced second moment accumulator
            self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype, shape=block_shape)
        else:
            # For scalar parameters, just use a single value
            self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype, shape=[1])
        
        # Add beta power accumulators
        try:
            type = core.VarDesc.VarType.DENSE_TENSOR
        except:
            type = core.VarDesc.VarType.LOD_TENSOR
            
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.9 if isinstance(self._beta1, (Variable, Value)) else self._beta1,
            shape=[1],
            type=type,
            device="cpu",
        )
        
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.999 if isinstance(self._beta2, (Variable, Value)) else self._beta2,
            shape=[1],
            type=type,
            device="cpu",
        )

    def _append_optimize_op(self, block, param_and_grad):
        """
        Add optimization operators for parameter.
        
        Args:
            block (Block): The block to add operators to.
            param_and_grad (tuple|dict): The parameter and gradient to optimize.
        """
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param = param_and_grad[0]

        # Whether we should do weight decay for the parameter
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
        master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
        lr = self._create_param_lr(param_and_grad)
        
        # Create the optimization op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            self.block_adam_mini_update(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                found_inf,
                _beta1,
                _beta2,
                self._epsilon,
                lr_ratio_,
                self._weight_decay,
                with_decay,
                find_master,
                self.block_size,
            )
            return None
        else:
            raise NotImplementedError("Not implemented for static graph mode.")

    def block_adam_mini_update(
        self,
        param,
        grad,
        learning_rate,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        skip_update,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        coeff,
        with_decay,
        multi_precision,
        block_size,
    ):
        """
        The core implementation of Block Adam-mini update algorithm.
        
        Args:
            param (Variable): Parameter to be updated.
            grad (Variable): Gradient of the parameter.
            learning_rate (Variable): Learning rate.
            moment1 (Variable): First moment accumulator.
            moment2 (Variable): Second moment accumulator (per block).
            beta1_pow (Variable): First order moment accumulator power.
            beta2_pow (Variable): Second order moment accumulator power.
            master_weight (Variable|None): Master weight for mixed precision training.
            skip_update (bool): Whether to skip this update.
            beta1 (float): Exponential decay rate for first moment estimator.
            beta2 (float): Exponential decay rate for second moment estimator.
            epsilon (float): Small value for numerical stability.
            lr_ratio (float): Learning rate ratio.
            coeff (float): Weight decay coefficient.
            with_decay (bool): Whether to apply weight decay.
            multi_precision (bool): Whether to use multi precision.
            block_size (int): Size of each block.
        """
        if skip_update:
            return
            
        if not with_decay:
            coeff = 0.0
            
        if not multi_precision:
            master_weight = None
            
        lr = learning_rate * lr_ratio
        
        if master_weight is not None:
            p = master_weight
        else:
            p = param
            
        # Apply weight decay
        p *= 1.0 - lr * coeff
        
        # Get shapes and calculate blocks
        shape = list(grad.shape)
        ndim = len(shape)
        size = int(np.prod(shape))
        num_blocks = max(1, int(size / block_size))
        
        # Update first moment (standard Adam)
        moment1 = beta1 * moment1 + (1.0 - beta1) * grad
        
        # For the second moment, we need to compute the mean square gradient per block
        flat_grad = grad.reshape([-1])
        
        # Create block-wise second moments
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, size)
            
            if start_idx < size:
                # Get block gradient
                block_grad = flat_grad[start_idx:end_idx]
                
                # Update block second moment
                block_mean_square = paddle.mean(block_grad * block_grad)
                
                if i == 0:
                    block_moment2 = beta2 * moment2[i] + (1.0 - beta2) * block_mean_square
                    moment2[i] = block_moment2
                else:
                    moment2[i] = beta2 * moment2[i] + (1.0 - beta2) * block_mean_square
        
        # Reshape parameter and first moment
        flat_p = p.reshape([-1])
        flat_moment1 = moment1.reshape([-1])
        
        # Apply update block-wise
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, size)
            
            if start_idx < size:
                # Get block corrected second moment
                block_moment2_corrected = moment2[i] / (1.0 - beta2_pow)
                
                # Calculate block denominator for update
                block_denom = paddle.sqrt(block_moment2_corrected) + epsilon
                
                # Apply block update to parameters
                block_moment1 = flat_moment1[start_idx:end_idx]
                block_update = block_moment1 / block_denom
                
                # Update parameters for this block
                flat_p[start_idx:end_idx] -= lr * block_update / (1.0 - beta1_pow)
        
        # Reshape parameter back to original shape
        p = flat_p.reshape(shape)
        
        # Update parameters and moments
        if master_weight is not None:
            master_weight[:] = p
            param[:] = p.astype(param.dtype)
        else:
            param[:] = p
            
        moment1[:] = moment1
        beta1_pow[:] = beta1 * beta1_pow
        beta2_pow[:] = beta2 * beta2_pow
        
        return
