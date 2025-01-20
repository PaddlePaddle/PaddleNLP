# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import Any, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from ....transformers import linear_utils

ColumnSequenceParallelLinear = linear_utils.ColumnSequenceParallelLinear
RowSequenceParallelLinear = linear_utils.RowSequenceParallelLinear

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        AllGatherOp,
        ReduceScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    AllGatherOp = None
    ReduceScatterOp = None
    mark_as_sequence_parallel_parameter = None

from paddlenlp.peft.tuners.tuners_utils import (
    BaseTunerLayer,  # , check_adapters_to_merg
)

from ....transformers.mc2_parallel_linear import (
    MC2ColumnParallelCoreLinear,
    MC2ColumnSeqParallelCoreLinear,
    MC2RowParallelCoreLinear,
    MC2RowSeqParallelCoreLinear,
)
from .lora_quick_layers import quick_lora
from .utils import rng_ctx


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Layer, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.LayerDict({})
        self.lora_A = nn.LayerDict({})
        self.lora_B = nn.LayerDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = paddle.nn.LayerDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.LayerDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            if self.lora_bias[adapter_name]:
                nn.init.zeros_(self.lora_B[adapter_name].bias)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                # embeddings are not supported at the moment, but still adding this for consistency
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def olora_init(self, adapter_name):
        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight
        bnb_param_type = get_bnb_param_type(orig_weight)
        dtype = orig_weight.dtype

        if bnb_param_type:
            # check without importing bitsandbytes and robust to bnb_4bit_quant_storage=float*
            weight_tensor = dequantize_module_weight(base_layer)
        elif dtype in [paddle.float32, paddle.float16, paddle.bfloat16]:
            weight_tensor = orig_weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(paddle.float32)
        Q, R = paddle.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()

        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        if bnb_param_type == "4bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                quant_type=orig_weight.quant_type,
                quant_storage=orig_weight.quant_storage,
                compress_statistics=orig_weight.compress_statistics,
                module=orig_weight.module,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        elif bnb_param_type == "8bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                requires_grad=orig_weight.requires_grad,
                has_fp16_weights=orig_weight.has_fp16_weights,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        else:
            weight_tensor = weight_tensor.to(dtype)
            base_layer.weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [paddle.float32, paddle.float16, paddle.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = transpose(weight.to(paddle.float32), self.fan_in_fan_out)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = paddle.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = paddle.diag(paddle.sqrt(Sr)) @ Uhr
        lora_B = Vr @ paddle.diag(paddle.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = transpose(weight.to(dtype), self.fan_in_fan_out)
        self.get_base_layer().weight.data = weight

    def corda_init(self, adapter_name, init_lora_weights):
        linear = self.get_base_layer()
        weight = linear.weight
        dtype = weight.dtype
        if dtype not in [paddle.float32, paddle.float16, paddle.bfloat16]:
            raise TypeError(
                "Please initialize CorDA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(paddle.float32)
        out_dim = weight.data.size(0)
        in_dim = weight.data.size(1)

        # Calculate WC from covariance matrix
        if not hasattr(linear, "eigens"):
            raise ValueError(
                "`eigens` attribute not found for layer, please run `preprocess_corda` first. "
                "More information can be found at examples/corda_finetuning/README.md."
            )
        eigens = linear.eigens
        U = eigens.U_WC
        S = eigens.S_WC
        V = eigens.V_WC
        r = self.r[adapter_name]

        # nan or inf check
        if paddle.isnan(S).any() or paddle.isinf(S).any():
            raise ValueError(
                "Invalid value found in matrix S. Please file an issue at https://github.com/huggingface/peft/issues."
            )
        if paddle.isnan(U).any() or paddle.isinf(U).any():
            raise ValueError(
                "Invalid value found in matrix U. Please file an issue at https://github.com/huggingface/peft/issues."
            )
        if paddle.isnan(V).any() or paddle.isinf(V).any():
            raise ValueError(
                "Invalid value found in matrix V. Please file an issue at https://github.com/huggingface/peft/issues."
            )

        # Sanity check
        if U.size(0) != out_dim or U.size(1) != r:
            raise ValueError(
                f"Matrix U size mismatch: {U.size()} vs. ({out_dim}, {r}). Please make sure the `lora_config` and "
                "`model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache "
                "in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank."
            )
        if S.size(0) != r:
            raise ValueError(
                f"Matrix S size mismatch: {S.size()} vs. ({r},). Please make sure the `lora_config` and `model` argument "
                "of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache in `preprocess_corda`, "
                "please make sure the cache is built with the same model and LoRA rank."
            )
        if V.size(0) != in_dim or V.size(1) != r:
            raise ValueError(
                f"Matrix V size mismatch: {V.size()} vs. ({in_dim}, {r}). Please make sure the `lora_config` and "
                "`model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache "
                "in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank."
            )

        # Apply alpha
        S /= self.scaling[adapter_name]

        # Init lora_A and lora_B weights
        lora_A = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        lora_B = U.mul(S.sqrt()).contiguous()
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def dora_init(self, adapter_name: str) -> None:
        if not self.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if self.ephemeral_gpu_offload:
            if lora_A.device.type in ["cuda", "xpu"]:
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type not in ["cuda", "xpu"]:
                    if is_xpu_available():
                        lora_B = lora_B.to("xpu")
                    else:
                        lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: paddle.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> paddle.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        paddle_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(paddle_result_dtype)

        return result


class LoRALinear(nn.Linear, LoraLayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_quick_lora: bool = False,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        pissa: bool = False,
        lora_use_mixer: bool = False,
        use_mora: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.use_mora = use_mora
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.pissa = pissa
        self.lora_use_mixer = lora_use_mixer

        # Actual trainable parameters
        if use_mora:  # reset the rank and create high rank matrix
            self.in_features = in_features
            self.out_features = out_features
            new_r = int(math.sqrt((in_features + out_features) * r) + 0.5)
            new_r = new_r // 2 * 2
            self.r = new_r
            self.lora_A = self.create_parameter(
                shape=[self.r, self.r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.cos = None
            self.sin = None
            # Count the number of tiles
            self.rb1 = self.in_features // self.r if self.in_features % self.r == 0 else self.in_features // self.r + 1
            self.rb2 = (
                self.out_features // self.r if self.out_features % self.r == 0 else self.out_features // self.r + 1
            )
            self.rope_init()
        else:
            self.lora_A = self.create_parameter(
                shape=[in_features, r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
            if self.lora_use_mixer:
                self.lora_AB = self.create_parameter(
                    shape=[r, r],
                    dtype=self._dtype,
                    is_bias=False,
                    default_initializer=nn.initializer.KaimingUniform(
                        negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                    ),
                )
            self.lora_B = self.create_parameter(
                shape=[r, out_features],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                    learning_rate=lora_plus_scale,
                ),
            )
        self.apply_pissa = False
        if use_mora or pissa:
            self.scaling = 1.0
        elif not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    def pissa_init(self, rank):
        weight = self.weight
        dtype = weight.dtype
        if dtype != paddle.float32:
            weight = weight.astype(paddle.float32)

        U, S, Vh = paddle.linalg.svd(weight.data, full_matrices=False)
        Ur = U[:, :rank]
        Sr = S[:rank]
        Vhr = Vh[:rank]

        lora_A = Ur @ paddle.diag(paddle.sqrt(Sr))
        lora_B = paddle.diag(paddle.sqrt(Sr)) @ Vhr
        self.lora_A.set_value(lora_A.astype(dtype))
        self.lora_B.set_value(lora_B.astype(dtype))
        res = weight.data - lora_A @ lora_B
        weight = res.astype(dtype)
        self.weight.set_value(weight)

    def rope_init(self):
        if self.cos is None or self.sin is None:
            inv_freq = 1.0 / (10000 ** (paddle.arange(0, self.r, 2, dtype=paddle.float32) / self.r))
            t = paddle.arange(self.rb1, dtype=paddle.float32)
            freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)
            emb = paddle.concat([freqs, freqs], axis=-1)
            self.cos = paddle.unsqueeze(paddle.cos(emb), axis=0).astype(self._dtype)
            self.sin = paddle.unsqueeze(paddle.sin(emb), axis=0).astype(self._dtype)

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def _apply_mora(self, x):
        r = self.r

        # Calculate grouping
        sum_inter = self.in_features // r

        # padding
        if self.in_features % r != 0:
            pad_size = r - self.in_features % r
            x = paddle.concat([x, x[..., :pad_size]], axis=-1)
            sum_inter += 1

        # reshape the input to apply RoPE
        in_x = x.reshape([*x.shape[:-1], sum_inter, r])

        # apply RoPE rotation
        rh_in_x = paddle.concat([-in_x[..., r // 2 :], in_x[..., : r // 2]], axis=-1)
        in_x = in_x * self.cos + rh_in_x * self.sin

        # matmul with high rank matrix
        out_x = in_x @ self.lora_A

        # reshape the output
        out_x = out_x.reshape([*x.shape[:-1], -1])[..., : self.out_features]
        if out_x.shape[-1] < self.out_features:
            repeat_time = self.out_features // out_x.shape[-1]
            if self.out_features % out_x.shape[-1] != 0:
                repeat_time += 1
            out_x = paddle.concat([out_x] * repeat_time, axis=-1)[..., : self.out_features]

        return out_x

    def get_delta_weight(self, lora_A=None, lora_B=None, lora_AB=None):
        # compute the delta weight，which is used to merge weights
        if self.lora_use_mixer:
            lora_A = lora_A if lora_A is not None else self.lora_A
            lora_B = lora_B if lora_B is not None else self.lora_B
            lora_AB = lora_AB if lora_AB is not None else self.lora_AB
            delta_weight = lora_A @ lora_AB @ lora_B * self.scaling
        elif self.use_mora:
            lora_A = lora_A if lora_A is not None else self.lora_A
            r = self.r
            # compute padding
            pad_size = r - self.in_features % r if self.in_features % r != 0 else 0
            # initialize weights
            w = paddle.zeros([self.in_features + pad_size, self.in_features], dtype=lora_A.dtype)

            # create the weights after rotation
            aw2 = paddle.concat([lora_A[:, r // 2 :], -lora_A[:, : r // 2]], axis=-1)
            # apply RoPE
            for i in range(self.rb1 - 1):
                w[i * r : (i + 1) * r, i * r : (i + 1) * r] = aw2 * self.sin[:, i] + lora_A * self.cos[:, i]
            # Process the last chunk that may be incomplete
            i = self.rb1 - 1
            w[i * r :, i * r :] = (aw2 * self.sin[:, i] + lora_A * self.cos[:, i])[:, : r - pad_size]
            # padding
            if pad_size > 0:
                w[i * r :, :pad_size] = (aw2 * self.sin[:, i] + lora_A * self.cos[:, i])[:, r - pad_size :]
            # reshape the weights
            if self.in_features < self.out_features:
                w = paddle.concat([w] * self.rb2, axis=0)[: self.out_features]
            else:
                w = w[: self.out_features]
            final_weight = w
            delta_weight = final_weight.T
        else:
            lora_A = lora_A if lora_A is not None else self.lora_A
            lora_B = lora_B if lora_B is not None else self.lora_B
            delta_weight = lora_A @ lora_B * self.scaling

        return delta_weight

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def forward(self, input: paddle.Tensor, *args, **kwargs):
        if not self.apply_pissa and self.pissa:
            self.pissa_init(self.r)
            self.apply_pissa = True
        if self.disable_lora or self.merged:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        elif self.use_quick_lora:
            # Use the quick lora implementation
            result = quick_lora(input, self.lora_A, self.lora_B, self.weight, self.bias, self.scaling)
        elif self.use_mora:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
            input = self.lora_dropout(input)
            mora_out = self._apply_mora(input)
            result += mora_out
        else:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
            if self.lora_use_mixer:
                result += (self.lora_dropout(input) @ self.lora_A @ self.lora_AB @ self.lora_B) * self.scaling
            else:
                result += (self.lora_dropout(input) @ self.lora_A @ self.lora_B) * self.scaling
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class RowParallelLoRALinear(RowParallelLinear, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        use_quick_lora: bool = False,
        pissa: bool = False,
        use_mora: bool = False,
        **kwargs
    ):
        RowParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")

        if pissa or use_mora:
            raise ValueError("Pissa or Mora is not supported in model parallel by now")

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

        # Actual trainable parameters
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_A = self.create_parameter(
                shape=[self.input_size_per_partition, r],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu")
                ),
            )
        self.lora_B = self.create_parameter(
            shape=[r, self.out_features],
            dtype=self._dtype,
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=lora_plus_scale,
            ),
        )

        self.lora_A.is_distributed = True
        self.lora_A.split_axis = 0
        self.lora_B.is_distributed = False
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(x, group=self.model_parallel_group)
        else:
            input_mp = x
        if self.disable_lora or self.merged:
            # x @ W : [bz, in_f / ws] ===> [bz, out_f]
            if MC2RowParallelCoreLinear is None:
                result_mp = F.linear(x=input_mp, weight=self.weight, name=self.name)
                output = mp_ops._mp_allreduce(
                    result_mp,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
            else:
                output = MC2RowParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)
            output = output + self.bias if self.bias is not None else output
        elif self.use_quick_lora:
            # Use the quick lora implementation
            result_mp = quick_lora(
                input_mp,
                self.lora_A,
                self.lora_B,
                self.weight,
                self.bias,
                self.scaling,
                is_row=True,
                group=self.model_parallel_group,
                world_size=self.world_size,
            )
            output = mp_ops._mp_allreduce(
                result_mp,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
        else:
            # x @ W : [bz, in_f / ws] ===> [bz, out_f]
            if MC2RowParallelCoreLinear is None:
                result_mp = F.linear(x=input_mp, weight=self.weight, name=self.name)
                output = mp_ops._mp_allreduce(
                    result_mp,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
            else:
                output = MC2RowParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)

            # x @ A: [bz, in_f/ ws] ===> [bz, r]
            input_mp = self.lora_dropout(input_mp) @ self.lora_A
            # all reduce to keep Lora B's gradient on different gpu consistent
            input_dup = mp_ops._mp_allreduce(
                input_mp,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
            #  @ B: [bz, r] ===> [bz, out_f]
            delta_mp = (input_dup @ self.lora_B) * self.scaling
            output += delta_mp
            output = output + self.bias if self.bias is not None else output
        return output

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class RowSequenceParallelLoRALinear(RowSequenceParallelLinear, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        use_quick_lora: bool = False,
        **kwargs
    ):
        RowSequenceParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

        # Actual trainable parameters
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_A = self.create_parameter(
                shape=[self.input_size_per_partition, r],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu")
                ),
            )
        self.lora_B = self.create_parameter(
            shape=[r, self.out_features],
            dtype=self._dtype,
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=lora_plus_scale,
            ),
        )

        self.lora_A.is_distributed = True
        self.lora_A.split_axis = 0
        self.lora_B.is_distributed = False
        mark_as_sequence_parallel_parameter(self.lora_B)
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        # TODO(@gexiao): support qlora
        return False  # self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(x, group=self.model_parallel_group)
        else:
            input_mp = x

        if MC2RowSeqParallelCoreLinear is None:
            output_parallel = self.linear(input_mp, self.weight, name=self._name)
            output_ = ReduceScatterOp.apply(output_parallel)
            result_mp = output_ + self.bias if self.bias is not None else output_
        else:
            output_ = MC2RowSeqParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)
            result_mp = output_ + self.bias if self.bias is not None else output_

        if not self.merged and not self.disable_lora:
            input_mp = self.lora_dropout(input_mp)
            # TODO(@gexiao): temporary workaround for deterministic calculation
            if True or MC2RowSeqParallelCoreLinear is None:
                input_mp = input_mp @ self.lora_A
                input_mp = ReduceScatterOp.apply(input_mp)
            else:
                input_mp = MC2RowSeqParallelCoreLinear.apply(input_mp, self.lora_A, self.model_parallel_group)
            delta_mp = (input_mp @ self.lora_B) * self.scaling
            result_mp += delta_mp
        return result_mp

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class ColumnParallelLoRALinear(ColumnParallelLinear, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        lora_A_weight_attr: Optional[paddle.ParamAttr] = None,
        use_quick_lora: bool = False,
        pissa: bool = False,
        use_mora: bool = False,
        **kwargs
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")

        if pissa or use_mora:
            raise ValueError("Pissa or Mora is not supported in model parallel by now")

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            attr=lora_A_weight_attr,
        )
        self.lora_A.is_distributed = False
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                    learning_rate=lora_plus_scale,
                ),
            )

        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        if self.disable_lora or self.merged:
            if MC2ColumnParallelCoreLinear is None:
                input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
                result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)
            else:
                res_mp = MC2ColumnParallelCoreLinear.apply(input, self.weight, self.model_parallel_group)
                result_mp = (res_mp + self.bias) if self.bias is not None else res_mp

        elif self.use_quick_lora:
            # Use the quick lora implementation
            input_mp = mp_ops._c_identity(input, group=self.model_parallel_group) if self.is_mp else input
            result_mp = quick_lora(
                input_mp,
                self.lora_A,
                self.lora_B,
                self.weight,
                self.bias,
                self.scaling,
                is_column=True,
                group=self.model_parallel_group,
                world_size=self.world_size,
            )
        else:
            if MC2ColumnParallelCoreLinear is None:
                input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
                result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)
            else:
                res_mp = MC2ColumnParallelCoreLinear.apply(input, self.weight, self.model_parallel_group)
                result_mp = (res_mp + self.bias) if self.bias is not None else res_mp

            input_a = self.lora_dropout(input) @ self.lora_A
            if MC2ColumnParallelCoreLinear is None:
                input_a_mp = mp_ops._c_identity(input_a, group=self.model_parallel_group)
                delta_mp = (input_a_mp @ self.lora_B) * self.scaling
            else:
                tmp = MC2ColumnParallelCoreLinear.apply(input_a, self.lora_B, self.model_parallel_group)
                delta_mp = tmp * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class ColumnSequenceParallelLoRALinear(ColumnSequenceParallelLinear, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        lora_A_weight_attr: Optional[paddle.ParamAttr] = None,
        use_quick_lora: bool = False,
        **kwargs
    ):
        ColumnSequenceParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            attr=lora_A_weight_attr,
        )
        self.lora_A.is_distributed = False
        mark_as_sequence_parallel_parameter(self.lora_A)

        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                    learning_rate=lora_plus_scale,
                ),
            )

        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        # TODO(@gexiao): support qlora
        return False  # self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if MC2ColumnSeqParallelCoreLinear is None:
            if self.is_mp:
                input_parallel = AllGatherOp.apply(x)
            else:
                input_parallel = x
            result_mp = self.linear(input_parallel, self.weight, self.bias, name=self._name)
        else:
            result_mp = MC2ColumnSeqParallelCoreLinear.apply(x, self.weight, self.model_parallel_group)
            if self.bias is not None:
                result_mp += self.bias

        if not self.merged and not self.disable_lora:
            input_a = self.lora_dropout(x) @ self.lora_A
            # TODO(@gexiao): temporary workaround for deterministic calculation
            if True or MC2ColumnSeqParallelCoreLinear is None:
                input_a = AllGatherOp.apply(input_a)
                delta_mp = (input_a @ self.lora_B) * self.scaling
            else:
                input_a = MC2ColumnSeqParallelCoreLinear.apply(input_a, self.lora_B, self.model_parallel_group)
                delta_mp = input_a * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class LoRAConv2D(nn.Conv2D):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs
    ):
        nn.Conv2D.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # Actual trainable parameters
        lora_A = nn.Conv2D(
            in_channels,
            r,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            weight_attr=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
            bias_attr=False,
        )
        self.lora_A = lora_A.weight
        self.lora_A_forward = lambda x: nn.Conv2D.__call__(lora_A, x)
        lora_B = nn.Conv2D(
            r,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            weight_attr=nn.initializer.Constant(value=0.0),
            bias_attr=False,
        )
        self.lora_B_forward = lambda x: nn.Conv2D.__call__(lora_B, x)
        self.lora_B = lora_B.weight
        self.scaling = lora_alpha / r

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        if self.bias is not None:
            self.bias.stop_gradient = True
        self.disable_lora = False

    def unmerge(self):
        if self.merged:
            weight_A = self.lora_A.cast(dtype=self.weight.dtype)
            weight_B = self.lora_B.cast(dtype=self.weight.dtype)
            if self.weight.shape[2:4] == [1, 1]:
                # conv2d 1x1
                delta_weight = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(
                    2
                ).unsqueeze(3) * self.scaling
            else:
                # conv2d 3x3
                delta_weight = (
                    F.conv2d(
                        weight_A.transpose([1, 0, 2, 3]),
                        weight_B,
                    ).transpose([1, 0, 2, 3])
                    * self.scaling
                )
            # Make sure that the weights are not merged
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            weight_A = self.lora_A.cast(dtype=self.weight.dtype)
            weight_B = self.lora_B.cast(dtype=self.weight.dtype)
            if self.weight.shape[2:4] == [1, 1]:
                # conv2d 1x1
                delta_weight = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(
                    2
                ).unsqueeze(3) * self.scaling
            else:
                # conv2d 3x3
                delta_weight = (
                    F.conv2d(
                        weight_A.transpose([1, 0, 2, 3]),
                        weight_B,
                    ).transpose([1, 0, 2, 3])
                    * self.scaling
                )
            # Merge the weights and mark it
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor, *args, **kwargs):
        previous_dtype = input.dtype
        result = super().forward(input)
        if not self.merged and not self.disable_lora:
            result += (
                self.lora_B_forward(self.lora_A_forward(self.lora_dropout(input.cast(dtype=self.lora_A.dtype))))
                * self.scaling
            )
        result = result.cast(dtype=previous_dtype)
        return result

    def extra_repr(self):
        main_str = "{_in_channels}, {_out_channels}, kernel_size={_kernel_size}"
        if self._stride != [1] * len(self._stride):
            main_str += ", stride={_stride}"
        if self._padding != 0:
            main_str += ", padding={_padding}"
        if self._padding_mode != "zeros":
            main_str += ", padding_mode={_padding_mode}"
        if self.output_padding != 0:
            main_str += ", output_padding={output_padding}"
        if self._dilation != [1] * len(self._dilation):
            main_str += ", dilation={_dilation}"
        if self._groups != 1:
            main_str += ", groups={_groups}"
        main_str += ", data_format={_data_format}, rank={r}, alpha={lora_alpha}"
        return main_str.format(**self.__dict__)
