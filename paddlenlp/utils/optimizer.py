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

import re

import paddle
from paddle import _C_ops, pir
from paddle.base import core, framework
from paddle.base.dygraph import base as imperative_base
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.base.libpaddle import DataType
from paddle.distributed import fleet
from paddle.optimizer.adamw import AdamW
from paddle.pir import Value

from paddlenlp.utils.log import logger

try:
    from .adamw_triton import adamw_triton
except:
    adamw_triton = None


from ..quantization.qat_utils import dequantize, quantize


class AdamWMini(AdamW):
    def __init__(
        self,
        named_parameters=None,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        use_lowprecision_moment=False,
        lr_ratio=None,
        apply_decay_param_fun=None,
        grad_clip=None,
        lazy_mode=False,
        multi_precision=False,
        amsgrad=False,
        dim=2048,
        n_heads=32,
        n_kv_heads=None,
        verbose=True,
        name=None,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_numel = self.dim * self.dim // self.n_heads
        self.verbose = verbose
        self.check_block_name = True
        self._already_create_accumulator = set()  # Initialize accumulator tracking set

        # Block naming patterns
        self.embd_names = {"embed", "embd", "wte"}
        self.output_names = {"lm_head", "output", "final_layer"}
        self.wqk_names = {"k_proj", "q_proj", "wq", "wk", "query", "key"}
        self.wv_names = {"v_proj", "wv", "value"}
        self.attn_proj_names = {"o_proj", "wo", "attn.proj"}
        self.mlp_names = {"feed_forward", "linear", "mlp"}
        self.adam_block_names = {"bias"}

        # Validation
        if not self.dim == int(self.dim):
            raise ValueError(f"Invalid dim value: {self.dim}")
        if not self.n_heads == int(self.n_heads):
            raise ValueError(f"Invalid n_heads value: {self.n_heads}")
        if not self.n_kv_heads == int(self.n_kv_heads):
            raise ValueError(f"Invalid n_kv_heads value: {self.n_kv_heads}")
        if not self.n_heads % self.n_kv_heads == 0:
            raise ValueError(f"n_heads {self.n_heads} must be divisible by n_kv_heads {self.n_kv_heads}")

        parameters = []
        for param_name, param in named_parameters:
            param_name = param_name.lower()
            param.name = param_name
            parameters.append(param)

        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            weight_decay=weight_decay,
            use_lowprecision_moment=use_lowprecision_moment,
            lr_ratio=lr_ratio,
            apply_decay_param_fun=apply_decay_param_fun,
            grad_clip=grad_clip,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision,
            amsgrad=amsgrad,
            name=name,
        )

    def _add_moments_pows(self, p):
        """Add moment accumulators with shapes based on block type."""
        name = p.name

        # Get accumulator data type
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype) and not self._use_lowprecision_moment:
            acc_dtype = DataType.FLOAT32 if in_pir_mode() else core.VarDesc.VarType.FP32

        # Add accumulators based on block type
        if any(adam_block_name in name for adam_block_name in self.adam_block_names):
            # Standard Adam for bias terms
            super()._add_moments_pows(p)
        elif any(wqk_name in name for wqk_name in self.wqk_names):
            # One accumulator per head for Q/K blocks
            total_size = paddle.numel(p)
            shape_moment1 = [total_size // self.head_numel, self.head_numel]
            shape_moment2 = [total_size // self.head_numel, 1]
            self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype, shape=shape_moment1)
            self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype, shape=shape_moment2)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                dtype=acc_dtype,
                fill_value=0.9 if isinstance(self._beta1, (Variable, Value)) else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.DENSE_TENSOR,
                device="cpu",
            )
            self._add_accumulator(
                name=self._beta2_pow_acc_str,
                param=p,
                dtype=acc_dtype,
                fill_value=0.999 if isinstance(self._beta2, (Variable, Value)) else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.DENSE_TENSOR,
                device="cpu",
            )
        elif (
            any(embd_name in name for embd_name in self.embd_names)
            or any(output_name in name for output_name in self.output_names)
            or any(wv_name in name for wv_name in self.wv_names)
            or any(mlp_name in name for mlp_name in self.mlp_names)
            or any(attn_proj_name in name for attn_proj_name in self.attn_proj_names)
        ):
            # One accumulator per neuron for other blocks
            if any(embd_name in name for embd_name in self.embd_names):
                shape = [p.shape[0], 1] if len(p.shape) > 1 else [1]
            else:
                shape = [1, p.shape[1]] if len(p.shape) > 1 else [1]

            self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
            self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype, shape=shape)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                dtype=acc_dtype,
                fill_value=0.9 if isinstance(self._beta1, (Variable, Value)) else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.DENSE_TENSOR,
                device="cpu",
            )
            self._add_accumulator(
                name=self._beta2_pow_acc_str,
                param=p,
                dtype=acc_dtype,
                fill_value=0.999 if isinstance(self._beta2, (Variable, Value)) else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.DENSE_TENSOR,
                device="cpu",
            )
        else:
            self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
            self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype, shape=[1])
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                dtype=acc_dtype,
                fill_value=0.9 if isinstance(self._beta1, (Variable, Value)) else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.DENSE_TENSOR,
                device="cpu",
            )
            self._add_accumulator(
                name=self._beta2_pow_acc_str,
                param=p,
                dtype=acc_dtype,
                fill_value=0.999 if isinstance(self._beta2, (Variable, Value)) else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.DENSE_TENSOR,
                device="cpu",
            )

    def _append_optimize_op(self, block, param_and_grad):
        """Implement optimization operations for different block types."""
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        param = param_and_grad[0]
        name = param.name

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        # Get moment accumulators
        moment1 = self._get_accumulator_master(self._moment1_acc_str, param)
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param)
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param)
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param)
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param.dtype)
        master_weight = self._master_weights[name] if find_master else None
        lr = self._create_param_lr(param_and_grad)

        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param)

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)
            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None

            self.adamw_python(
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
                name,
            )
            return None
        else:
            raise NotImplementedError("Not implemented yet.")

    def adamw_python(
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
        name,
    ):
        if skip_update:
            return
        if not with_decay:
            coeff = 0.0
        if "norm" in name or "ln" in name or "bias" in name:
            coeff = 0.0
        if not multi_precision:
            master_weight = None

        if any(adam_block_name in name for adam_block_name in self.adam_block_names):
            _, _, _, _, _, _, _ = _C_ops.adamw_(
                param,
                grad,
                learning_rate,
                moment1,
                moment2,
                None,
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
                self._lazy_mode,
                1000,
                multi_precision,
                False,
                self._amsgrad,
            )

        else:
            lr = learning_rate * lr_ratio
            if master_weight is not None:
                p = master_weight
            else:
                p = param
            p *= 1.0 - lr * coeff

            # Block-specific updates with per-block learning rates
            if any(wqk_name in name for wqk_name in self.wqk_names):
                # Q/K blocks: reshape and compute per-head learning rates
                grad_reshaped = paddle.reshape(grad, [-1, self.head_numel])
                mom1 = paddle.reshape(moment1, [-1, self.head_numel])
                mom2 = moment2  # Already shaped correctly

                # Compute per-head second moment
                mom2_update = paddle.mean(grad_reshaped * grad_reshaped, axis=1, keepdim=True)
                # Update moments with correct beta values
                mom1 = mom1 * beta1 + (1.0 - beta1) * grad_reshaped
                mom2 = mom2 * beta2 + (1.0 - beta2) * mom2_update

                # Compute adaptive learning rate
                denom = mom2.sqrt() / ((1.0 - beta2_pow).sqrt()) + epsilon

                # Apply updates
                update = (mom1 / denom) * (-(lr / (1.0 - beta1_pow)))
                p += paddle.reshape(update, param.shape)

            elif (
                any(embd_name in name for embd_name in self.embd_names)
                or any(output_name in name for output_name in self.output_names)
                or any(wv_name in name for wv_name in self.wv_names)
                or any(mlp_name in name for mlp_name in self.mlp_names)
                or any(attn_proj_name in name for attn_proj_name in self.attn_proj_names)
            ):
                mom1 = moment1
                mom2 = moment2  # Already shaped correctly

                mom1 = mom1 * beta1 + (1.0 - beta1) * grad

                if any(embd_name in name for embd_name in self.embd_names):
                    mom2 = mom2 * beta2 + (1.0 - beta2) * (grad * grad).mean(axis=1, keepdim=True)
                else:
                    mom2 = mom2 * beta2 + (1.0 - beta2) * (grad * grad).mean(axis=0, keepdim=True)

                denom = mom2.sqrt() / ((1.0 - beta2_pow).sqrt()) + epsilon
                p += (mom1 / denom) * (-(lr / (1.0 - beta1_pow)))

            else:
                # Other blocks
                mom1 = moment1
                mom2 = moment2  # Already shaped correctly

                mom1 = mom1 * beta1 + (1.0 - beta1) * grad
                mom2 = mom2 * beta2 + (1.0 - beta2) * (grad * grad).mean()

                denom = mom2.sqrt() / ((1.0 - beta2_pow).sqrt()) + epsilon
                p += (mom1 / denom) * (-(lr / (1.0 - beta1_pow)))

            # Update param in-place
            if master_weight is not None:
                master_weight[:] = p
                param[:] = p.astype(param.dtype)
            else:
                param[:] = p

            # Update accumulators in-place
            moment1[:] = mom1
            moment2[:] = mom2
            beta1_pow[:] = beta1 * beta1_pow[:]
            beta2_pow[:] = beta2 * beta2_pow[:]

        return None

    def _count_block(self):
        """Count the number of each block type for logging."""
        if not self.verbose:
            return

        counts = {
            "embedding": 0,
            "output": 0,
            "query/key": 0,
            "value": 0,
            "attention_proj": 0,
            "mlp": 0,
        }

        for name in self._already_create_accumulator:
            if "bias" in name:
                continue
            if any(embd_name in name for embd_name in self.embd_names):
                counts["embedding"] += 1
            if any(output_name in name for output_name in self.output_names):
                counts["output"] += 1
            if any(wqk_name in name for wqk_name in self.wqk_names):
                counts["query/key"] += 1
            if any(wv_name in name for wv_name in self.wv_names):
                counts["value"] += 1
            if any(attn_proj_name in name for attn_proj_name in self.attn_proj_names):
                counts["attention_proj"] += 1
            if any(mlp_name in name for mlp_name in self.mlp_names):
                counts["mlp"] += 1

        logger.info("\nAdam-mini found blocks:")
        logger.info(f"- {counts['embedding']} embedding layers")
        logger.info(f"- {counts['output']} output layers")
        logger.info(f"- {counts['query/key']} Query and Key layers")
        logger.info(f"- {counts['value']} Value layers")
        logger.info(f"- {counts['attention_proj']} Attention projection layers")
        logger.info(f"- {counts['mlp']} MLP layers\n")

        # Print warnings for missing blocks
        if counts["embedding"] == 0:
            logger.warning("Warning: No embedding layers found")
        if counts["output"] == 0:
            logger.warning("Warning: No output layers found (ignore if using weight tying)")
        if counts["query/key"] == 0:
            logger.warning("Warning: No Query/Key layers found")
        if counts["value"] == 0:
            logger.warning("Warning: No Value layers found")
        if counts["attention_proj"] == 0:
            logger.warning("Warning: No attention projection layers found")
        if counts["mlp"] == 0:
            logger.warning("Warning: No MLP layers found")
        if sum(counts.values()) == 0:
            logger.warning("Warning: No Transformer blocks found")

    def _create_accumulators(self, block, parameters):
        """Create accumulators for parameters."""
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_moments_pows(master_p)
                self._already_create_accumulator.add(p.name)
                continue
            if self._is_dtype_fp16_or_bf16(p.dtype) and not self._multi_precision:
                logger.warning(
                    "Accumulating with FP16 or BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )
            self._add_moments_pows(p)
            self._already_create_accumulator.add(p.name)

        if self.check_block_name:
            self._count_block()
            self.check_block_name = False


class AdamWCustom(AdamW):
    def __init__(self, quantization_config, tensorwise_offload_optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_scale_mapping = {}
        for p in self._param_groups:
            if "quantization_linear" in p.name and "w_1" in p.name:
                self.quant_scale_mapping[p.name.replace("w_1", "w_0")] = p
        self.quantization_config = quantization_config
        self._hcg = fleet.get_hybrid_communicate_group()
        self.mp_group = self._hcg.get_model_parallel_group()
        self.tensorwise_offload_optimizer = tensorwise_offload_optimizer

    def _add_moments_pows(self, p, moment_dtype=core.VarDesc.VarType.FP32):
        acc_dtype = p.dtype

        self._add_accumulator(self._moment1_acc_str, p, dtype=moment_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=moment_dtype)
        try:
            type = core.VarDesc.VarType.DENSE_TENSOR
        except:
            type = core.VarDesc.VarType.LOD_TENSOR
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=(0.9 if isinstance(self._beta1, (Variable, Value)) else self._beta1),
            shape=[1],
            type=type,
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=(0.999 if isinstance(self._beta2, (Variable, Value)) else self._beta2),
            shape=[1],
            type=type,
        )

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                if self._use_lowprecision_moment:
                    if p.name in self.quant_scale_mapping:
                        p_scale = self.quant_scale_mapping[p.name]
                        if str(p_scale.dtype) == "paddle.float16":
                            moment_dtype = core.VarDesc.VarType.FP16
                        elif str(p_scale.dtype) == "paddle.bfloat16":
                            moment_dtype = core.VarDesc.VarType.BF16
                    else:
                        if str(p.dtype) == "paddle.float16":
                            moment_dtype = core.VarDesc.VarType.FP16
                        elif str(p.dtype) == "paddle.bfloat16":
                            moment_dtype = core.VarDesc.VarType.BF16
                else:
                    moment_dtype = core.VarDesc.VarType.FP32

                self._add_moments_pows(master_p, moment_dtype)
                self._already_create_accumulator.add(p.name)

            elif self._is_dtype_fp16_or_bf16(p.dtype) and not self._multi_precision:
                raise NotImplementedError("AdamWCustom only support AMP training")
            else:
                self._add_moments_pows(p)
                self._already_create_accumulator.add(p.name)
            if self.tensorwise_offload_optimizer:
                self.offload_optim(p)

    def _create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            var_name = self._gen_master_weight_var_name(param)
            if param.name in self.quant_scale_mapping:
                quant_scale = self.quant_scale_mapping[param.name]
                if self.quantization_config.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                    var = dequantize(
                        param,
                        quant_scale,
                        "weight",
                        self.quantization_config.weight_quantize_algo,
                        self.quantization_config,
                        apply_hadamard=self.quantization_config.apply_hadamard,
                        side="left",
                    ).astype("float32")
                else:
                    raise NotImplementedError(
                        f"Unknown weight_quantize_algo {self.quantization_config.weight_quantize_algo}"
                    )
            else:
                var = paddle.cast(param, "float32")
            var.name = var_name
            self._master_weights[param.name] = var
        return var

    def _is_dtype_fp16_or_bf16(self, dtype):
        """
        check the dtype is fp16 or the dtype is bf16
        :param dtype: instance of core.VarDesc.VarType
        :return: True if dtype is one of fp16 or bf16, False otherwise
        """
        if dtype == paddle.int8 or dtype == paddle.float8_e4m3fn:
            return True
        assert isinstance(
            dtype, (core.VarDesc.VarType, core.DataType)
        ), "The dtype should be an instance of core.VarDesc.VarType or core.DataType."
        if isinstance(dtype, core.VarDesc.VarType):
            return dtype == core.VarDesc.VarType.FP16 or dtype == core.VarDesc.VarType.BF16
        else:
            return dtype == core.DataType.FLOAT16 or dtype == core.DataType.BFLOAT16

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        if self.tensorwise_offload_optimizer:
            self.reload_optim(param)

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
        master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
        if param.name in self.quant_scale_mapping:
            quant_scale = self.quant_scale_mapping[param.name]
        else:
            quant_scale = None
        lr = self._create_param_lr(param_and_grad)
        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            skip_update_param = quant_scale is not None
            apply_adamw = self.adamw_custom if adamw_triton is None else adamw_triton
            apply_adamw(
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
                skip_update_param,
            )
            if skip_update_param:
                if param.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                    if "parallel_quantization_linear" not in param.name:
                        group = None
                    elif param.weight_quantize_algo in ["a8w8linear", "a8w4linear"] and "row" in param.name:
                        group = None
                    else:
                        group = self.mp_group
                    param[:], quant_scale[:] = quantize(
                        x=master_weight.astype(quant_scale.dtype),
                        weight_quantize_algo=self.quantization_config.weight_quantize_algo,
                        tensor_type="weight",
                        quantization_config=self.quantization_config,
                        side="left",
                        apply_hadamard=self.quantization_config.apply_hadamard,
                        group=group,
                    )
                else:
                    raise NotImplementedError(
                        f"Please check your weight_quantize_algo {self.quantization_config.weight_quantize_algo}."
                    )
            if self.tensorwise_offload_optimizer:
                self.offload_optim(param)

            return None
        else:
            raise NotImplementedError("Not implemented yet.")

    def adamw_custom(
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
        skip_update_param,
    ):
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

        p *= 1.0 - lr * coeff
        moment_dtype = moment1.dtype
        mom1 = moment1.astype("float32")
        mom2 = moment2.astype("float32")

        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * grad * grad
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        p += (mom1 / denom) * (-(lr / (1.0 - beta1_pow)))

        if master_weight is not None:
            master_weight[:] = p
            if not skip_update_param:
                param[:] = p.astype(param.dtype)
        else:
            param[:] = p
        moment1[:] = mom1.astype(moment_dtype)
        moment2[:] = mom2.astype(moment_dtype)
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
        return

    def offload_optim(self, p):
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype)
        if find_master:
            self._master_weights[p.name] = self._master_weights[p.name].pin_memory()
            target_name = self._master_weights[p.name].name
        else:
            target_name = p.name
        for name in [self._moment1_acc_str, self._moment2_acc_str]:
            if self._name is not None:
                name = self._name + "_" + name
            self._accumulators[name][target_name] = self._accumulators[name][target_name].pin_memory()

    def reload_optim(self, p):
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype)
        if find_master:
            self._master_weights[p.name] = self._master_weights[p.name].cuda()
            target_name = self._master_weights[p.name].name
        else:
            target_name = p.name
        for name in [self._moment1_acc_str, self._moment2_acc_str]:
            if self._name is not None:
                name = self._name + "_" + name
            self._accumulators[name][target_name] = self._accumulators[name][target_name].cuda()


class AdamWLoRAPro(AdamW):
    def __init__(self, scaling_factor=2.0, x_mode="zero", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert scaling_factor is not None
        if x_mode not in ["zero", "sylvester", "symmetry"]:
            raise ValueError(
                f"Invalid x_mode value: {x_mode}, " f"mode should be in ['zero', 'sylvester', 'symmetry']"
            )
        self.scaling_factor = scaling_factor
        self.x_mode = x_mode

    def _solve_sylvester(self, A, B, C, X=None):
        if A.dtype in [paddle.bfloat16, paddle.float16]:
            A = A.to("float32")
            B = B.to("float32")
            C = C.to("float32")
        B = -B
        m = tuple(B.shape)[-1]
        n = tuple(A.shape)[-1]
        R, U = paddle.linalg.eig(x=A)
        S, V = paddle.linalg.eig(x=B)

        CV = C @ V

        U_real, U_imag = paddle.real(U), paddle.imag(U)
        CV_real, CV_imag = paddle.real(CV), paddle.imag(CV)

        n_dim = U_real.shape[0]

        block_top = paddle.concat([U_real, -U_imag], axis=1)  # (n, 2n)
        block_bot = paddle.concat([U_imag, U_real], axis=1)  # (n, 2n)
        A_block = paddle.concat([block_top, block_bot], axis=0)  # (2n, 2n)
        B_block = paddle.concat([CV_real, CV_imag], axis=0)  # (2n, m)

        F_block = paddle.linalg.solve(A_block, B_block)  # [F_real; F_imag]

        F_real = F_block[:n_dim, :]
        F_imag = F_block[n_dim:, :]
        F = paddle.complex(F_real, F_imag)

        W = R[..., :, None] - S[..., None, :]
        Y = F / W
        try:
            V_inv = paddle.linalg.inv(V)
        except RuntimeError:
            # Add regularization to handle singular matrices
            epsilon = 1e-6 * paddle.mean(paddle.abs(V))
            V_reg = V + epsilon * paddle.eye(V.shape[-1])
            V_inv = paddle.linalg.inv(V_reg)
        X = U[..., :n, :n] @ Y[..., :n, :m] @ V_inv[..., :m, :m]

        if all(paddle.isreal(x.flatten()[0]) for x in [A, B, C]):
            return paddle.real(X)
        else:
            return X

    @imperative_base.no_grad
    @framework.non_static_only
    def step(self) -> None:
        """
        Execute the optimizer and update parameters once.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> a = paddle.rand([2,13], dtype="float32")
                >>> linear = paddle.nn.Linear(13, 5)
                >>> # This can be any optimizer supported by dygraph.
                >>> opt = paddle.optimizer.AdamW(learning_rate = 0.01,
                ...                             parameters = linear.parameters())
                >>> out = linear(a)
                >>> out.backward()
                >>> opt.step()
                >>> opt.clear_grad()
        """
        if paddle.base.dygraph.base.in_to_static_mode():
            self._declarative_step()
            return

        if not isinstance(self._parameter_list[0], dict):
            param_id_to_idx = {id(param): idx for idx, param in enumerate(self._parameter_list)}

            lora_params = {}
            for idx, param in enumerate(self._parameter_list):
                name = getattr(param, "name", f"param_{idx}")
                match = re.match(r"lo_ra_linear_(\d+)\.w_(\d+)", name)
                if match:
                    layer_num = int(match.group(1))
                    weight_type = match.group(2)
                    if layer_num not in lora_params:
                        lora_params[layer_num] = {}
                    lora_params[layer_num][weight_type] = param

            for layer_num, weights in lora_params.items():
                if "1" in weights and "2" in weights:
                    param_B = weights["1"]
                    param_A = weights["2"]

                    idx_B = param_id_to_idx[id(param_B)]
                    idx_A = param_id_to_idx[id(param_A)]

                    if param_A._grad_ivar() is not None and param_B._grad_ivar() is not None:
                        A = param_A.detach()
                        B = param_B.detach()
                        grad_A = param_A._grad_ivar()
                        grad_B = param_B._grad_ivar()

                        delta = 1e-08
                        AA_T = A @ A.T
                        B_TB = B.T @ B
                        AA_T_inv = paddle.linalg.pinv(AA_T + delta * paddle.eye(num_rows=AA_T.shape[0]))
                        B_TB_inv = paddle.linalg.pinv(B_TB + delta * paddle.eye(num_rows=B_TB.shape[0]))

                        if self.x_mode == "sylvester":
                            X = self._solve_sylvester(
                                B_TB, AA_T, -(1 / self.scaling_factor**2) * B_TB_inv @ grad_A @ A.T
                            )
                        elif self.x_mode == "symmetry":
                            X = -0.5 * (1 / self.scaling_factor**2) * B_TB_inv @ B.T @ grad_B @ AA_T
                        else:  # zero mode
                            X = paddle.zeros(shape=(B_TB_inv.shape[0], B_TB_inv.shape[0]))

                        X = X.clone().detach().cast(A.dtype)

                        new_grad_A = (1 / self.scaling_factor**2) * B_TB_inv @ grad_A + X @ A
                        new_grad_B = (1 / self.scaling_factor**2) * (
                            (paddle.eye(num_rows=B.shape[0]) - B @ B_TB_inv @ B.T) @ grad_B @ AA_T_inv
                        ) - B @ X

                        self._parameter_list[idx_A]._grad_ivar()[:] = new_grad_A
                        self._parameter_list[idx_B]._grad_ivar()[:] = new_grad_B

            params_grads = []
            for param in self._parameter_list:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    if framework.in_dygraph_mode():
                        if (
                            hasattr(grad_var, "is_selected_rows")
                            and grad_var.is_selected_rows()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "AdamW don't support weight_decay with sparse parameters, please set it to None."
                            )
                    else:
                        if (
                            hasattr(grad_var, "_is_sparse")
                            and grad_var._is_sparse()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "AdamW don't support weight_decay with sparse parameters, please set it to None."
                            )
                    params_grads.append((param, grad_var))

                    self._apply_optimize(loss=None, startup_program=None, params_grads=params_grads)
        else:
            raise NotImplementedError("AdamWLoRAPro does not support parameter groups")
