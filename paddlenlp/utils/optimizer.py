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
import warnings

import paddle
from paddle import _C_ops, pir
from paddle.base import core, framework
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.base.libpaddle import DataType
from paddle.optimizer.adamw import AdamW
from paddle.pir import Value

try:
    from paddlenlp_kernel.triton.optimizer import adamw_16bit_moment
except:
    adamw_16bit_moment = None

from ..quantization.qat_utils import dequantize_channelwise, quantize_channelwise


class AdamWMini(AdamW):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        parameters=None,
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
        if any(adam_block_name in name.lower() for adam_block_name in self.adam_block_names):
            # Standard Adam for bias terms
            super()._add_moments_pows(p)
        elif any(wqk_name in name.lower() for wqk_name in self.wqk_names):
            # One accumulator per head for Q/K blocks
            shape = [-1, self.head_numel]
            self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype, shape=shape)
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
        elif (
            any(embd_name in name.lower() for embd_name in self.embd_names)
            or any(output_name in name.lower() for output_name in self.output_names)
            or any(wv_name in name.lower() for wv_name in self.wv_names)
            or any(mlp_name in name.lower() for mlp_name in self.mlp_names)
            or any(attn_proj_name in name.lower() for attn_proj_name in self.attn_proj_names)
        ):
            # One accumulator per neuron for other blocks
            shape = [p.shape[0], 1] if len(p.shape) > 1 else [1]
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
        if not multi_precision:
            master_weight = None

        if any(adam_block_name in name.lower() for adam_block_name in self.adam_block_names):
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
            if any(wqk_name in name.lower() for wqk_name in self.wqk_names):
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
                update = (mom1 / denom.reshape([-1, 1])) * (-(lr / (1.0 - beta1_pow)))
                p += paddle.reshape(update, param.shape)

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
            name = name.lower()
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

        print("\nAdam-mini found blocks:")
        print(f"- {counts['embedding']} embedding layers")
        print(f"- {counts['output']} output layers")
        print(f"- {counts['query/key']} Query and Key layers")
        print(f"- {counts['value']} Value layers")
        print(f"- {counts['attention_proj']} Attention projection layers")
        print(f"- {counts['mlp']} MLP layers\n")

        # Print warnings for missing blocks
        if counts["embedding"] == 0:
            print("Warning: No embedding layers found")
        if counts["output"] == 0:
            print("Warning: No output layers found (ignore if using weight tying)")
        if counts["query/key"] == 0:
            print("Warning: No Query/Key layers found")
        if counts["value"] == 0:
            print("Warning: No Value layers found")
        if counts["attention_proj"] == 0:
            print("Warning: No attention projection layers found")
        if counts["mlp"] == 0:
            print("Warning: No MLP layers found")
        if sum(counts.values()) == 0:
            print("Warning: No Transformer blocks found")

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
                warnings.warn(
                    "Accumulating with FP16 or BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )
            self._add_moments_pows(p)
            self._already_create_accumulator.add(p.name)

        if self.check_block_name:
            self._count_block()
            self.check_block_name = False


class AdamWCustom(AdamW):
    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
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
        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            self.adamw_custom(
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
            )
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
        mom1 = moment1
        mom2 = moment2

        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * grad * grad
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        p += (mom1 / denom) * (-(lr / (1.0 - beta1_pow)))
        if master_weight is not None:
            master_weight[:] = p
            param[:] = p.astype(param.dtype)
        else:
            param[:] = p
        moment1[:] = mom1
        moment2[:] = mom2
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
        return


class AdamW_16Bit(AdamW):
    def _add_moments_pows(self, p, moment_dtype=core.VarDesc.VarType.FP32):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            acc_dtype = DataType.FLOAT32 if in_pir_mode() else core.VarDesc.VarType.FP32

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
                if str(p.dtype) == "paddle.float16":
                    moment_dtype = core.VarDesc.VarType.FP16
                elif str(p.dtype) == "paddle.bfloat16":
                    moment_dtype = core.VarDesc.VarType.BF16

                self._add_moments_pows(master_p, moment_dtype)
                self._already_create_accumulator.add(p.name)
                continue
            if self._is_dtype_fp16_or_bf16(p.dtype) and not self._multi_precision:
                warnings.warn(
                    "Accumulating with FP16 or BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )
            self._add_moments_pows(p)
            self._already_create_accumulator.add(p.name)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
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
        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            apply_adamw = self.adamw_16bit_moment if adamw_16bit_moment is None else adamw_16bit_moment
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
            )
            return None
        else:
            raise NotImplementedError("Not implemented yet.")

    def adamw_16bit_moment(
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
        mom1 = moment1
        mom2 = moment2

        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * grad * grad
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        p += (mom1 / denom) * (-(lr / (1.0 - beta1_pow))).astype("float32")
        if master_weight is not None:
            master_weight[:] = p
            param[:] = p.astype(param.dtype)
        else:
            param[:] = p
        moment1[:] = mom1.astype(moment_dtype)
        moment2[:] = mom2.astype(moment_dtype)
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
        return


class AdamWQweight(AdamW):
    def __init__(self, quantization_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_scale_mapping = {}
        for p in self._param_groups:
            if "quantization_linear" in p.name and "w_1" in p.name:
                self.quant_scale_mapping[p.name.replace("w_1", "w_0")] = p

        self.quantization_config = quantization_config

    def _create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            var_name = self._gen_master_weight_var_name(param)
            if param.name in self.quant_scale_mapping:
                quant_scale = self.quant_scale_mapping[param.name]
                var = dequantize_channelwise(
                    param, quant_scale, apply_hadamard=self.quantization_config.apply_hadamard
                ).astype("float32")
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
        if dtype == paddle.int8:
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
            self.adamw_custom(
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
                quant_scale,
            )
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
        quant_scale,
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
        mom1 = moment1
        mom2 = moment2

        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * grad * grad
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        p += (mom1 / denom) * (-(lr / (1.0 - beta1_pow)))

        if master_weight is not None:
            master_weight[:] = p
            if quant_scale is None:
                param[:] = p.astype(param.dtype)
            else:
                if p.shape[1] == param.shape[0]:
                    bit_length = 8
                elif p.shape[1] / 2 == param.shape[0]:
                    bit_length = 4
                param[:], quant_scale[:] = quantize_channelwise(
                    p.astype("bfloat16"), apply_hadamard=self.quantization_config.apply_hadamard, bit_length=bit_length
                )
        else:
            param[:] = p
        moment1[:] = mom1
        moment2[:] = mom2
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
        return
