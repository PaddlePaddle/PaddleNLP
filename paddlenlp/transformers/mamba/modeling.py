# coding=utf-8
# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
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
"""Paddle MAMBA model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.nn import CrossEntropyLoss

from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)

from ...utils.initializer import constant_, kaiming_uniform_, normal_, uniform_, zeros_
from ..activations import ACT2FN
from ..model_outputs import ModelOutput
from ..model_utils import PretrainedModel
from .configuration import MambaConfig

try:
    from mamba_ssm_paddle.ops.selective_scan_interface import (
        mamba_inner_fn,
        selective_scan_fn,
    )
    from mamba_ssm_paddle.ops.triton.selective_state_update import (
        selective_state_update,
    )
except ImportError:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

try:
    from mamba_ssm_paddle.ops.causal_conv1d_interface import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)

from paddlenlp.utils.log import logger

########################################################################################################################

_CHECKPOINT_FOR_DOC = "state-spaces/mamba-130m-hf"
_CONFIG_FOR_DOC = "MambaConfig"

__all__ = [
    "MambaMixer",
    "MambaBlock",
    "MambaModel",
    "MambaPretrainedModel",
    "MambaForCausalLM",
]


class MambaCache:
    """
    Arguments:
        config: MambaConfig
        batch_size: int
        dtype: paddle.dtype

    Attributes:
        seqlen_offset: int
        dtype: paddle.dtype
        conv_states: Dict[int, paddle.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, paddle.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
        self,
        config: MambaConfig,
        batch_size: int,
        dtype: paddle.dtype = paddle.float16,
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.config = config
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: paddle.zeros([batch_size, intermediate_size, conv_kernel_size], dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: paddle.zeros([batch_size, intermediate_size, ssm_state_size], dtype=dtype)
            for i in range(config.num_hidden_layers)
        }

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(self.config.num_hidden_layers):
            # In-place ops prevent breaking the static address
            self.conv_states[layer_idx].zero_()
            self.ssm_states[layer_idx].zero_()
        self.seqlen_offset = 0


class MambaMixer(nn.Layer):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1D(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias_attr=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias_attr=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias_attr=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias_attr=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = paddle.arange(1, self.ssm_state_size + 1, dtype=paddle.float32)[None, :]
        A = A.expand([self.intermediate_size, -1]).contiguous()

        self.A_log = self.create_parameter(
            shape=A.shape,
            default_initializer=nn.initializer.Assign(paddle.log(A)),
        )
        self.D = self.create_parameter(
            shape=[
                self.intermediate_size,
            ],
            default_initializer=nn.initializer.Constant(1),
        )
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=config.use_bias)
        self.use_bias = config.use_bias
        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/JunnYu/mamba/tree/paddle-1.2.2/#installation. "
            )

    def cuda_kernels_forward(self, hidden_states: paddle.Tensor, cache: Optional[MambaCache] = None):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose([0, 2, 1])

        if self.training and cache is None:  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.cast("float32") if self.use_bias else None,
                -paddle.exp(self.A_log.cast("float32")),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.cast("float32"),
                delta_bias=self.dt_proj.bias.cast("float32"),
                delta_softplus=True,
                is_paddle_linear=True,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, axis=1)

            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.reshape([self.conv1d.weight.shape[0], self.conv1d.weight.shape[2]])
            if cache is not None and cache.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache is not None:
                    conv_states = nn.functional.pad(
                        hidden_states,
                        (self.conv_kernel_size - hidden_states.shape[-1], 0),
                        data_format="NCL",
                    )
                    cache.conv_states[self.layer_idx].copy_(conv_states.cast(cache.dtype), False)
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose([0, 2, 1]))
            time_step, B, C = paddle.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], axis=-1
            )
            discrete_time_step = self.dt_proj.weight.t() @ time_step.transpose([0, 2, 1])

            A = -paddle.exp(self.A_log.cast("float32"))
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.cast("float32") if hasattr(self.dt_proj, "bias") else None
            if cache is not None and cache.seqlen_offset > 0:
                scan_outputs = selective_state_update(
                    cache.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose([0, 2, 1]),
                    C.transpose([0, 2, 1]),
                    self.D.cast("float32"),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache is not None:
                    cache.ssm_states[self.layer_idx].copy_(ssm_state.cast(cache.dtype), False)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose([0, 2, 1]))
        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache: Optional[MambaCache] = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose([0, 2, 1])                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, axis=1)

        # 2. Convolution sequence transformation
        if cache is not None:
            ssm_state = cache.ssm_states[self.layer_idx].clone()
            if cache.seqlen_offset > 0:
                conv_state = cache.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = paddle.roll(conv_state, shifts=-1, axis=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache.conv_states[self.layer_idx].copy_(conv_state.cast(cache.dtype), False)
                hidden_states = paddle.sum(conv_state * self.conv1d.weight[:, 0, :], axis=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).cast(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0),
                    data_format="NCL",
                )
                cache.conv_states[self.layer_idx].copy_(conv_state.cast(cache.dtype), False)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = paddle.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                dtype=dtype,
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose([0, 2, 1]))
        time_step, B, C = paddle.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], axis=-1
        )
        discrete_time_step = self.dt_proj(time_step)                                          # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose([0, 2, 1])  # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -paddle.exp(self.A_log.cast("float32"))                                             # [intermediate_size, ssm_state_size]
        discrete_A = paddle.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])        # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].cast("float32")       # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].cast("float32")

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]         # [batch, intermediade_size, ssm_state]
            scan_output = paddle.matmul(ssm_state.cast(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = paddle.stack(scan_outputs, axis=-1)                                 # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache is not None:
            cache.ssm_states[self.layer_idx].copy_(ssm_state.cast(cache.dtype), False)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose([0, 2, 1]))             # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(self, hidden_states, cache: Optional[MambaCache] = None):
        if is_fast_path_available:
            return self.cuda_kernels_forward(hidden_states, cache)
        return self.slow_forward(hidden_states, cache)


class MambaRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = self.create_parameter(
            shape=[
                hidden_size,
            ],
            default_initializer=nn.initializer.Constant(1),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.cast(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.cast(input_dtype)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


class MambaBlock(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, cache: Optional[MambaCache] = None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.cast(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.cast(paddle.float32)

        hidden_states = self.mixer(hidden_states, cache=cache)
        hidden_states = residual + hidden_states
        return hidden_states


class MambaPretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MambaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["MambaBlock"]
    supports_gradient_checkpointing = True

    @classmethod
    def _get_name_mappings(cls, config: MambaConfig) -> List[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embeddings.weight"],
            ["norm_f.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [f"layers.{layer_index}.norm.weight"],
                [f"layers.{layer_index}.mixer.A_log"],
                [f"layers.{layer_index}.mixer.D"],
                [f"layers.{layer_index}.mixer.conv1d.weight"],
                [f"layers.{layer_index}.mixer.in_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mixer.x_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mixer.dt_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mixer.out_proj.weight", None, "transpose"],
            ]
            layer_mappings.append([f"layers.{layer_index}.mixer.dt_proj.bias"])

            if config.use_conv_bias:
                layer_mappings.append([f"layers.{layer_index}.mixer.conv1d.bias"])
            if config.use_bias:
                layer_mappings.append([f"layers.{layer_index}.mixer.in_proj.bias"])
                layer_mappings.append([f"layers.{layer_index}.mixer.out_proj.bias"])
            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)

        # base-model prefix "MambaModel"
        if "MambaModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "backbone." + mapping[0]
                mapping[1] = "backbone." + mapping[1]
            if not config.tie_word_embeddings:
                model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = paddle.exp(
                paddle.rand((self.config.intermediate_size,), dtype="float32").cast(paddle.get_default_dtype())
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clip(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + paddle.log(-paddle.expm1(-dt))
            with paddle.no_grad():
                module.dt_proj.bias.copy_(inv_dt, False)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    kaiming_uniform_(p, a=math.sqrt(5))
                    with paddle.no_grad():
                        p.copy_(p / math.sqrt(self.config.num_layers), False)


@dataclass
class MambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[paddle.Tensor] = None
    cache: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class MambaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    cache: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None


MAMBA_START_DOCSTRING = r"""

    This model inherits from [`PretrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [paddle.nn.Layer](https://pypaddle.org/docs/stable/nn.html#paddle.nn.Layer) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MambaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PretrainedModel.from_pretrained`] method to load the model weights.
"""

MAMBA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`paddle.Tensor` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary.

            If `cache.seqlen_offset>0`, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PretrainedTokenizer.encode`] and
            [`PretrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        cache (`MambaCache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache` is returned and can be used to quickly generate the next logits.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class MambaModel(MambaPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.LayerList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.enable_recompute = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        cache: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.enable_recompute and self.training and use_cache:
            use_cache = False

        if cache is None and use_cache:
            cache = MambaCache(self.config, inputs_embeds.shape[0], dtype=inputs_embeds.dtype)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.enable_recompute and self.training and not hidden_states.stop_gradient:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = recompute(
                    create_custom_forward(mixer_block),
                    hidden_states,
                    cache,
                )
            else:
                hidden_states = mixer_block(hidden_states, cache=cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache=cache if use_cache else None,
            hidden_states=all_hidden_states,
        )


class MambaForCausalLM(MambaPretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaModel(config)
        if self.config.tie_word_embeddings:
            self.lm_head = lambda x: paddle.matmul(x, self.backbone.embeddings.weight, transpose_y=True)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return None
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if self.config.tie_word_embeddings:
            return None
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache"] = outputs.get("cache", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=True,
        cache: Optional[MambaCache] = None,
        **kwargs,
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache"] = cache
        model_inputs["use_cache"] = use_cache
        return model_inputs

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        cache: Optional[MambaCache] = None,
        labels: Optional[paddle.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.cast(self.get_input_embeddings().weight.dtype)).cast("float32")

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :]
            # shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.reshape([-1, logits.shape[-1]]),
                labels.reshape(
                    [
                        -1,
                    ]
                ),
            )

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache=mamba_outputs.cache,
            hidden_states=mamba_outputs.hidden_states,
        )
