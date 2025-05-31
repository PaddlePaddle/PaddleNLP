
import math
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddle.utils import try_import

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

from .. import linear_utils
from ..linear_utils import Linear
from ..segment_parallel_utils import ReshardLayer
from ..utils import caculate_llm_per_token_flops
from .configuration import (
    GEMMA2_PRETRAINED_INIT_CONFIGURATION,
    GEMMA2_PRETRAINED_RESOURCE_FILES_MAP,
    Gemma2Config,
)

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def rms_norm_fused(x_in, w, eps):
    fused_ln = try_import("fused_ln")
    return fused_ln.fused_rms_norm(x_in, w, eps)[0]


def assign_kv_heads(num_kv_heads: int, num_gpus: int):
    """
    Assign kv heads to different GPUs in the Tensor Parallel Setup

    Examples:
        assign_kv_heads(num_kv_heads=1, num_gpus=2): [[0], [0]]
        assign_kv_heads(num_kv_heads=2, num_gpus=2): [[0], [1]]
        assign_kv_heads(num_kv_heads=4, num_gpus=2): [[0,1], [2,3]]
        assign_kv_heads(num_kv_heads=1, num_gpus=4): [[0],[0],[0],[0]]
        assign_kv_heads(num_kv_heads=2, num_gpus=4): [[0],[0],[1],[1]]
        assign_kv_heads(num_kv_heads=4, num_gpus=4): [[0],[1],[2],[3]]
    """
    assignment_list = [[] for _ in range(num_gpus)]
    if num_kv_heads > num_gpus:
        num_heads_per_card = num_kv_heads // num_gpus
        for i in range(num_gpus):
            for j in range(num_heads_per_card):
                assignment_list[i].append(i * num_heads_per_card + j)
    else:
        num_card_per_heads = num_gpus // num_kv_heads
        for i in range(num_kv_heads):
            for j in range(num_card_per_heads):
                assignment_list[i * num_card_per_heads + j].append(i)
    return assignment_list


def build_alibi_tensor(
    bool_attention_mask: Tensor, num_heads: int, dtype: paddle.dtype, tensor_parallel_degree=1
) -> Tensor:
    attention_mask = bool_attention_mask.astype("float32")
    batch_size, seq_length = attention_mask.shape[0], attention_mask.shape[-1]
    slopes = paddle.to_tensor(_get_interleave(num_heads), dtype="float32")
    alibi = slopes.unsqueeze(axis=[1, 2]) * paddle.arange(seq_length, dtype="float32").unsqueeze(axis=[0, 1]).expand(
        [num_heads, -1, -1]
    )
    alibi = alibi.reshape(shape=(1, num_heads, 1, seq_length)).expand([batch_size, -1, -1, -1])
    return paddle.cast(alibi, dtype)


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    shape = x.shape
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(hidden_states, n_rep, axis=1). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states.unsqueeze(-2).tile([1, 1, 1, n_rep, 1])
    return hidden_states.reshape([batch, slen, num_key_value_heads * n_rep, head_dim])


def parallel_matmul(x: Tensor, y, tensor_parallel_output=True, transpose_y=False):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_parallel_degree > 1

    if is_fleet_init and tensor_parallel_degree > 1 and y_is_distributed:
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=transpose_y)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=transpose_y)
        return logits

def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    sequence_parallel=False,
    reshard_layer=None,
    attn_dropout_prob=0.0,
    trainer_mode=False,
    sliding_window=None,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    if hasattr(config, "attn_logit_softcapping") and config.attn_logit_softcapping > 0:
        attn_logit_softcapping = config.attn_logit_softcapping
    else:
        attn_logit_softcapping = None

    if hasattr(config, "use_flash_attention") and config.use_flash_attention and flash_attention:

        version = paddle.version.full_version
        if version != "0.0.0" and version <= "2.5.2":
            if alibi is not None:
                raise ValueError("Flash Attention doesn't support alibi")
            attn_output, attn_weights = flash_attention(
                query_states,
                key_states,
                value_states,
                causal=True,
                dropout=attn_dropout_prob,
                return_softmax=output_attentions,
            )
        else:
            if alibi is not None:
                alibi = alibi.reshape([bsz, num_heads, 1, -1])
                attention_mask = attention_mask.cast(alibi.dtype) + alibi
            
            if sliding_window is not None and sliding_window > 0:
                window_mask = paddle.ones((q_len, kv_seq_len), dtype=query_states.dtype)
                window_mask = paddle.triu(window_mask, diagonal=1-sliding_window)
                window_mask = paddle.tril(window_mask, diagonal=0)
                window_mask = 1.0 - window_mask
                window_mask = window_mask * paddle.finfo(query_states.dtype).min
                window_mask = window_mask.unsqueeze([0, 1])  # [1, 1, q_len, kv_seq_len]
                
                if attention_mask is not None:
                    attention_mask = attention_mask + window_mask
                else:
                    attention_mask = window_mask
            
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
            attn_weights = None

        if reshard_layer is not None:
            attn_output = reshard_layer(
                attn_output,
                split_axis=1,
                concat_axis=2,
            )
            assert (
                config.sep_parallel_degree > 1 and q_len % config.sep_parallel_degree == 0
            ), f"q_len:{q_len}, config.sep_parallel_degree:{config.sep_parallel_degree}"
            q_len = q_len // config.sep_parallel_degree
            num_heads = num_heads * config.sep_parallel_degree

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output
    else:
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        if hasattr(config, "query_pre_attn_scalar") and config.query_pre_attn_scalar > 0:
            query_states = query_states * (config.query_pre_attn_scalar ** -0.5)

        attn_weights = paddle.matmul(query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2]))
        if alibi is not None:
            alibi = alibi.reshape([bsz, num_heads, 1, -1])
            attn_weights = attn_weights + alibi

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if reshard_layer is not None:
            attention_mask = None

        if sliding_window is not None and sliding_window > 0:
            window_mask = paddle.ones((q_len, kv_seq_len), dtype=attn_weights.dtype)
            window_mask = paddle.triu(window_mask, diagonal=1-sliding_window)
            window_mask = paddle.tril(window_mask, diagonal=0)
            window_mask = 1.0 - window_mask
            window_mask = window_mask * paddle.finfo(attn_weights.dtype).min
            window_mask = window_mask.unsqueeze([0, 1])  # [1, 1, q_len, kv_seq_len]
            
            if attention_mask is not None:
                attention_mask = attention_mask + window_mask
            else:
                attention_mask = window_mask

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)
        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        
        if attn_logit_softcapping is not None:
            attn_weights = attn_logit_softcapping * F.tanh(attn_weights / attn_logit_softcapping)
            
        if not paddle.in_dynamic_mode():
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        else:
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        attn_weights = F.dropout(attn_weights, attn_dropout_prob, training=trainer_mode)
        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        if reshard_layer is not None:
            attn_output = reshard_layer(
                attn_output,
                split_axis=1,
                concat_axis=2,
            )
            q_len = q_len // config.sep_parallel_degree
            num_heads = num_heads * config.sep_parallel_degree

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output

def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class Gemma2RMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        if hasattr(self.config, "use_fused_rms_norm") and self.config.use_fused_rms_norm:
            return rms_norm_fused(x, self.weight, self.variance_epsilon)

        output = self._norm(x.astype(paddle.float32)).astype(x.dtype)
        return output * self.weight


class Gemma2RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))

    def forward(self, x, seq_len=None):
        t = paddle.arange(seq_len, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat([freqs, freqs], axis=-1)
        return (emb.cos()[None, :, None, :].cast(dtype=x.dtype), emb.sin()[None, :, None, :].cast(dtype=x.dtype))

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Gemma2MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel_degree = getattr(config, "tensor_parallel_degree", 1)

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

        if config.tensor_parallel_degree > 1:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
            self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
            self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Gemma2Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma2Config, layerwise_recompute: bool = False):
        super().__init__()

        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.seq_length = config.seq_length if hasattr(config, "seq_length") else config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        self.sliding_window = config.sliding_window if hasattr(config, "sliding_window") else None

        self.kv_indices = None
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = getattr(config, "recompute_granularity", "full_attn")
        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            if self.num_key_value_heads % config.tensor_parallel_degree == 0:
                self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree
            else:
                self.kv_indices = paddle.to_tensor(
                    assign_kv_heads(self.num_key_value_heads, config.tensor_parallel_degree)[
                        config.tensor_parallel_rank
                    ]
                )

        self.use_fused_rope = getattr(config, "use_fused_rope", False)
        if self.use_fused_rope:
            if "gpu" not in paddle.device.get_device() or fused_rotary_position_embedding is None:
                warnings.warn(
                    "Enable fuse rope in the config, but fuse rope is not available. "
                    "Will disable fuse rope. Try using latest gpu version of Paddle."
                )
                self.use_fused_rope = False

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

        if config.tensor_parallel_degree > 1:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.config.num_attention_heads * self.head_dim,
                has_bias=getattr(config, "attention_bias", False),
                gather_output=False,
            )
            if self.kv_indices is None:
                self.k_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    has_bias=getattr(config, "attention_bias", False),
                    gather_output=False,
                )
                self.v_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    has_bias=getattr(config, "attention_bias", False),
                    gather_output=False,
                )
            else:
                self.k_proj = Linear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    bias_attr=False,
                )
                self.v_proj = Linear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    bias_attr=False,
                )

        else:
            self.q_proj = Linear(
                self.hidden_size,
                self.config.num_attention_heads * self.head_dim,
                bias_attr=False,
            )
            self.k_proj = Linear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                bias_attr=False,
            )
            self.v_proj = Linear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                bias_attr=False,
            )

        if config.tensor_parallel_degree > 1:
            self.o_proj = RowParallelLinear(
                self.config.num_attention_heads * self.head_dim,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            self.o_proj = Linear(
                self.config.num_attention_heads * self.head_dim,
                self.hidden_size,
                bias_attr=False,
            )
        self.rotary_emb = Gemma2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.reshard_layer = None
        if getattr(config, "sep_parallel_degree", 1) > 1:
            assert self.num_key_value_heads % config.sep_parallel_degree == 0
            assert self.num_heads % config.sep_parallel_degree == 0
            self.reshard_layer = ReshardLayer()

        self.config = config

    def forward(
        self,
        hidden_states,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        alibi: Optional[paddle.Tensor] = None,
    ):
        """Input shape: Batch x Time x Channel"""
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.reshard_layer is not None:
            if self.sequence_parallel:
                assert self.seq_length % self.config.sep_parallel_degree == 0
                query_states = paddle.reshape(
                    query_states,
                    [-1, self.seq_length // self.config.sep_parallel_degree, self.num_heads * self.head_dim],
                )
                key_states = paddle.reshape(
                    key_states,
                    [-1, self.seq_length // self.config.sep_parallel_degree, self.num_heads * self.head_dim],
                )
                value_states = paddle.reshape(
                    value_states,
                    [-1, self.seq_length // self.config.sep_parallel_degree, self.num_heads * self.head_dim],
                )
            query_states = self.reshard_layer(
                query_states,
                split_axis=2,
                concat_axis=1,
            )
            key_states = self.reshard_layer(
                key_states,
                split_axis=2,
                concat_axis=1,
            )
            value_states = self.reshard_layer(
                value_states,
                split_axis=2,
                concat_axis=1,
            )
            query_states = paddle.reshape(
                query_states, [0, self.seq_length, -1, self.head_dim]
            )  # [bs, seq_len, num_head/k, head_dim], k is sep degree
            key_states = paddle.reshape(key_states, [0, self.seq_length, -1, self.head_dim])
            value_states = paddle.reshape(value_states, [0, self.seq_length, -1, self.head_dim])
        else:
            if self.sequence_parallel:
                target_query_shape = [-1, self.seq_length, self.num_heads, self.head_dim]
                target_key_value_shape = [-1, self.seq_length, self.num_key_value_heads, self.head_dim]
            else:
                target_query_shape = [0, 0, self.num_heads, self.head_dim]
                target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
            query_states = query_states.reshape(shape=target_query_shape)
            key_states = key_states.reshape(shape=target_key_value_shape)
            value_states = value_states.reshape(shape=target_key_value_shape)

        kv_seq_len = key_states.shape[-3]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        if getattr(self.config, "rope", True):
            if self.reshard_layer is not None:
                batch_size, seq_length, _, _ = query_states.shape
                position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))
            if self.use_fused_rope and fused_rotary_position_embedding is not None:
                assert past_key_value is None, "fuse rotary not support cache kv for now"
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                paddle_version = float(paddle.__version__[:3])
                if ((paddle_version != 0.0) and (paddle_version <= 2.6)) and (
                    self.num_heads != self.num_key_value_heads
                ):
                    query_states, _, _ = fused_rotary_position_embedding(
                        query_states,
                        None,
                        None,
                        sin=sin,
                        cos=cos,
                        position_ids=position_ids,
                        use_neox_rotary_style=False,
                    )
                    key_states, _, _ = fused_rotary_position_embedding(
                        key_states,
                        None,
                        None,
                        sin=sin,
                        cos=cos,
                        position_ids=position_ids,
                        use_neox_rotary_style=False,
                    )
                else:
                    query_states, key_states, _ = fused_rotary_position_embedding(
                        query_states,
                        key_states,
                        v=None,
                        sin=sin,
                        cos=cos,
                        position_ids=position_ids,
                        use_neox_rotary_style=False,
                    )
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.kv_indices is not None:
            key_states = paddle.index_select(key_states, self.kv_indices, axis=2)
            value_states = paddle.index_select(value_states, self.kv_indices, axis=2)
            key_states = paddle.broadcast_to(key_states, query_states.shape)
            value_states = paddle.broadcast_to(value_states, query_states.shape)
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        ):
            outputs = recompute(
                scaled_dot_product_attention,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                alibi,
                self.sequence_parallel,
                reshard_layer=self.reshard_layer,
                use_reentrant=getattr(self.config, "recompute_use_reentrant", False),
                attn_dropout_prob=self.attention_dropout,
                trainer_mode=self.training,
                sliding_window=self.sliding_window,
            )
        else:
            outputs = scaled_dot_product_attention(
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                alibi,
                self.sequence_parallel,
                reshard_layer=self.reshard_layer,
                attn_dropout_prob=self.attention_dropout,
                trainer_mode=self.training,
                sliding_window=self.sliding_window,
            )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs

class Gemma2DecoderLayer(nn.Layer):
    def __init__(self, config: Gemma2Config, layerwise_recompute: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma2Attention(config, layerwise_recompute=layerwise_recompute)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config)
        self.post_attention_layernorm = Gemma2RMSNorm(config)
        self.config = config
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = getattr(config, "recompute_granularity", "full_attn")
        self.sequence_parallel = getattr(config, "sequence_parallel", False)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        alibi: Optional[paddle.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        has_gradient = not hidden_states.stop_gradient
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "full_attn"
        ):
            hidden_states, self_attn_weights, present_key_value = recompute(
                self.self_attn,
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                use_cache=use_cache,
                alibi=alibi,
                use_reentrant=getattr(self.config, "recompute_use_reentrant", False),
            )
        else:
            hidden_states = self.self_attn(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                use_cache=use_cache,
                alibi=alibi,
            )

        if output_attentions:
            hidden_states, self_attn_weights = hidden_states
            outputs = (hidden_states, self_attn_weights)
        else:
            outputs = (hidden_states,)

        if use_cache:
            present_key_value = outputs[-1]
            outputs = outputs[:-1] + (present_key_value,)
        else:
            present_key_value = None

        hidden_states = outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        has_gradient = not hidden_states.stop_gradient
        if self.enable_recompute and self.layerwise_recompute and has_gradient:
            hidden_states = recompute(
                self.mlp,
                hidden_states,
                use_reentrant=getattr(self.config, "recompute_use_reentrant", False),
            )
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class Gemma2PretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Gemma2Config
    base_model_prefix = "gemma2"
    pretrained_init_configuration = GEMMA2_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = GEMMA2_PRETRAINED_RESOURCE_FILES_MAP
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, layer):
        """Initialize the weights."""
        if isinstance(layer, (nn.Linear, nn.Embedding, Linear)):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )
        elif isinstance(layer, Gemma2RMSNorm):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(paddle.ones_like(layer.weight))


class Gemma2Model(Gemma2PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Gemma2DecoderLayer`]
    """

    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.max_position_embeddings = config.max_position_embeddings
        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        self.sep_parallel_degree = getattr(config, "sep_parallel_degree", 1)
        self.tensor_parallel_degree = getattr(config, "tensor_parallel_degree", 1)
        self.tensor_parallel_rank = getattr(config, "tensor_parallel_rank", 0)
        self.recompute_granularity = getattr(config, "recompute_granularity", "full_attn")
        self.layerwise_recompute = getattr(config, "recompute", False)
        self.use_recompute = self.layerwise_recompute
        self.embed_dim = config.hidden_size

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = ColumnParallelLinear(
                config.vocab_size,
                config.hidden_size,
                has_bias=False,
                gather_output=True,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.LayerList(
            [Gemma2DecoderLayer(config, layerwise_recompute=self.layerwise_recompute) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Gemma2RMSNorm(config)

        self.gradient_checkpointing = False
        self.config = config

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
        combined_attention_mask = None
        if input_shape[1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            dtype = combined_attention_mask.dtype if combined_attention_mask is not None else paddle.float32
            expanded_attn_mask = _expand_2d_mask(attention_mask, dtype=dtype, tgt_length=input_shape[1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[1]

        if position_ids is None:
            position_ids = paddle.arange(past_key_values_length, seq_length + past_key_values_length, dtype="int64")
            position_ids = position_ids.unsqueeze(0).expand([batch_size, seq_length])

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length + past_key_values_length), dtype=paddle.bool)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), past_key_values_length
        )

        hidden_states = inputs_embeds

        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [] if output_attentions else None
        next_decoder_cache = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                if all_hidden_states is not None:
                    all_hidden_states.append(hidden_states)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                if next_decoder_cache is not None:
                    next_decoder_cache.append(layer_outputs[2 if output_attentions else 1])

            if output_attentions:
                if all_self_attns is not None:
                    all_self_attns.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states = tuple(all_hidden_states)
        if all_self_attns is not None:
            all_self_attns = tuple(all_self_attns)
        if next_decoder_cache is not None:
            next_decoder_cache = tuple(next_decoder_cache)
            
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@register_base_model
class Gemma2ForCausalLM(Gemma2PretrainedModel):
    """
    Gemma2 Model with a `language modeling` head on top.
    """

    def __init__(self, config):
        super().__init__(config)
        self.gemma2 = Gemma2Model(config)
        self.vocab_size = config.vocab_size
        self.final_logit_softcapping = config.final_logit_softcapping if hasattr(config, "final_logit_softcapping") else None
        self.tensor_parallel_degree = getattr(config, "tensor_parallel_degree", 1)
        self.tensor_parallel_rank = getattr(config, "tensor_parallel_rank", 0)
        self.sequence_parallel = getattr(config, "sequence_parallel", False)

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear

        if config.tensor_parallel_degree > 1:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                has_bias=False,
                gather_output=True,
            )
        else:
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias_attr=False)

    def get_input_embeddings(self):
        return self.gemma2.embed_tokens

    def set_input_embeddings(self, value):
        self.gemma2.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.gemma2 = decoder

    def get_decoder(self):
        return self.gemma2

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PretrainedTokenizer.encode`] and
                [`PretrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
                model's internal embedding lookup matrix.
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithCrossAttentions`.

        Examples:
            .. code-block::

                >>> from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

                >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
                >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

                >>> inputs = tokenizer("I enjoy walking with my cute dog", return_tensors="pd")
                >>> outputs = model(**inputs, labels=inputs["input_ids"])
                >>> loss = outputs.loss
                >>> logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gemma2(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0:
            logits = self.final_logit_softcapping * F.tanh(logits / self.final_logit_softcapping)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape([-1, self.vocab_size]), shift_labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    sequence_parallel=False,
    reshard_layer=None,
    attn_dropout_prob=0.0,
    trainer_mode=False,
    sliding_window=None,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    if hasattr(config, "attn_logit_softcapping") and config.attn_logit_softcapping > 0:
        attn_logit_softcapping = config.attn_logit_softcapping
    else:
        attn_logit_softcapping = None

    if config.use_flash_attention and flash_attention:

        version = paddle.version.full_version
        if version != "0.0.0" and version <= "2.5.2":
            if alibi is not None:
                raise ValueError("Flash Attention doesn't support alibi")
            attn_output, attn_weights = flash_attention(
                query_states,
                key_states,
                value_states,
                causal=True,
                dropout=attn_dropout_prob,
                return_softmax=output_attentions,
            )
        else:
            if alibi is not None:
                alibi = alibi.reshape([bsz, num_heads, 1, -1])
                attention_mask = attention_mask.cast(alibi.dtype) + alibi
            
            if sliding_window is not None and sliding_window > 0:
                window_mask = paddle.ones((q_len, kv_seq_len), dtype=query_states.dtype)
                window_mask = paddle.triu(window_mask, diagonal=1-sliding_window)
                window_mask = paddle.tril(window_mask, diagonal=0)
                window_mask = 1.0 - window_mask
                window_mask = window_mask * paddle.finfo(query_states.dtype).min
                window_mask = window_mask.unsqueeze([0, 1])  # [1, 1, q_len, kv_seq_len]
                
                if attention_mask is not None:
                    attention_mask = attention_mask + window_mask
                else:
                    attention_mask = window_mask
            
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
            attn_weights = None

        if reshard_layer is not None:
            attn_output = reshard_layer(
                attn_output,
                split_axis=1,
                concat_axis=2,
            )
            assert (
                config.sep_parallel_degree > 1 and q_len % config.sep_parallel_degree == 0
            ), f"q_len:{q_len}, config.sep_parallel_degree:{config.sep_parallel_degree}"
            q_len = q_len // config.sep_parallel_degree
            num_heads = num_heads * config.sep_parallel_degree

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output
    else:
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2]))
        if alibi is not None:
            alibi = alibi.reshape([bsz, num_heads, 1, -1])
            attn_weights = attn_weights + alibi

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if reshard_layer is not None:
            attention_mask = None

        if sliding_window is not None and sliding_window > 0:
            window_mask = paddle.ones((q_len, kv_seq_len), dtype=attn_weights.dtype)
            window_mask = paddle.triu(window_mask, diagonal=1-sliding_window)
            window_mask = paddle.tril(window_mask, diagonal=0)
            window_mask = 1.0 - window_mask
            window_mask = window_mask * paddle.finfo(attn_weights.dtype).min
            window_mask = window_mask.unsqueeze([0, 1])  # [1, 1, q_len, kv_seq_len]
            
            if attention_mask is not None:
                attention_mask = attention_mask + window_mask
            else:
                attention_mask = window_mask

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)
        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        
        if attn_logit_softcapping is not None:
            attn_weights = attn_logit_softcapping * F.tanh(attn_weights / attn_logit_softcapping)
            
        if not paddle.in_dynamic_mode():
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        else:
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        attn_weights = F.dropout(attn_weights, attn_dropout_prob, training=trainer_mode)
        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        if reshard_layer is not None:
            attn_output = reshard_layer(
                attn_output,
                split_axis=1,
                concat_axis=2,
            )
            q_len = q_len // config.sep_parallel_degree
            num_heads = num_heads * config.sep_parallel_degree

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class Gemma2RMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        if hasattr(self.config, "use_fused_rms_norm") and self.config.use_fused_rms_norm:
            return rms_norm_fused(x, self.weight, self.variance_epsilon)

        output = self._norm(x.astype(paddle.float32)).astype(x.dtype)
        return output * self.weight


class Gemma2RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))

    def forward(self, x, seq_len=None):
        t = paddle.arange(seq_len, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat([freqs, freqs], axis=-1)
        return (emb.cos()[None, :, None, :].cast(dtype=x.dtype), emb.sin()[None, :, None, :].cast(dtype=x.dtype))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):

    if position_ids is None:
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Gemma2MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel_degree = getattr(config, "tensor_parallel_degree", 1)

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            self.gate_proj = Linear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
                name="gate_proj",
            )
            self.up_proj = Linear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
                name="up_proj",
            )
            self.down_proj = Linear(
                self.intermediate_size,
                self.hidden_size,
                scatter_input=False,
                has_bias=False,
                name="down_proj",
            )
        else:
            self.gate_proj = Linear(
                self.hidden_size,
                self.intermediate_size,
                parallel="col",
                has_bias=False,
                name="gate_proj",
            )
            self.up_proj = Linear(
                self.hidden_size,
                self.intermediate_size,
                parallel="col",
                has_bias=False,
                name="up_proj",
            )
            self.down_proj = Linear(
                self.intermediate_size,
                self.hidden_size,
                parallel="row",
                has_bias=False,
                name="down_proj",
            )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
