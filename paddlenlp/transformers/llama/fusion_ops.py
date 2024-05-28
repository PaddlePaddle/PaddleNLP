# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import paddle.nn.functional as F

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


from paddle.utils import try_import

from paddlenlp.utils.tools import get_env_device

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None
try:
    if get_env_device() in ["npu", "gcu"]:
        from paddle.base import core

        for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
            if lib.endswith(".so"):
                paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(lib)
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None


def fusion_rope(query_states, key_states, value_states, hidden_states, position_ids, past_key_value, rotary_emb):
    if get_env_device() != "gcu":
        assert past_key_value is None, "fuse rotary not support cache kv for now"
    batch_size, seq_length, num_heads, head_dim = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = key_states.shape
    if get_env_device() != "gcu":
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
    if get_env_device() == "npu":
        query_states = core.eager._run_custom_op("fused_rope", query_states, cos, sin)[0]
        key_states = core.eager._run_custom_op("fused_rope", key_states, cos, sin)[0]
    elif get_env_device() == "gcu":
        cos_sin = rotary_emb.get_fused_cos_sin(value_states, seq_len=kv_seq_len)
        query_states, key_states = core.eager._run_custom_op(
            "fused_rotary_embedding_gcu", query_states, key_states, cos_sin, position_ids, True
        )
    else:
        # paddle version > 2.6 or develop support q and k/v with different num_heads
        paddle_version = float(paddle.__version__[:3])
        if ((paddle_version != 0.0) and (paddle_version <= 2.6)) and (num_heads != num_key_value_heads):
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
    return query_states, key_states


def rms_norm_fused(x_in, w, eps):
    fused_ln = try_import("fused_ln")
    return fused_ln.fused_rms_norm(x_in, w, eps)[0]


def fusion_rms_norm(hidden_states, weight, variance_epsilon):
    if get_env_device() == "npu":
        return core.eager._run_custom_op("rms_norm_npu", hidden_states, weight, variance_epsilon)[0]
    elif get_env_device() == "gcu":
        return core.eager._run_custom_op("rms_norm_gcu", hidden_states, weight, variance_epsilon)[0]
    elif get_env_device() == "xpu":
        try:
            import paddle_xpu_nn  # noqa: F821

            return paddle_xpu_nn.xpu_rms_norm(hidden_states, weight, variance_epsilon)[0]
        except ImportError:
            raise NotImplementedError(
                f"Implementation of fused_rms_norm is not available on {get_env_device()}. Please install paddle_xpu to use this feature"
            )
    return rms_norm_fused(hidden_states, weight, variance_epsilon)


def fusion_flash_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    sequence_parallel=False,
    reshard_layer=None,
    npu_is_casual=False,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape
    version = paddle.version.full_version
    if version != "0.0.0" and version <= "2.5.2":
        if alibi is not None:
            raise ValueError("Flash Attention doesn't support alibi")
        attn_output, attn_weights = flash_attention(
            query_states,
            key_states,
            value_states,
            causal=True,
            return_softmax=output_attentions,
        )
    else:
        if alibi is not None:
            alibi = alibi.reshape([bsz, num_heads, 1, -1])
            attention_mask = attention_mask.cast(alibi.dtype) + alibi
        if get_env_device() == "npu":
            attn_output = core.eager._run_custom_op(
                "flash_attention_npu",
                query_states,
                key_states,
                value_states,
                None,
                attention_mask,
                0.0,
                attention_mask is None,
                True,
                False,
                npu_is_casual,
            )[0]
        elif get_env_device() == "gcu":
            attn_output = core.eager._run_custom_op(
                "fused_sdp_flash_attention_gcu",
                query_states,
                key_states,
                value_states,
                attention_mask,
                0.0,
                attention_mask is None,
                True,
            )[0]
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
        attn_weights = None

    if reshard_layer is not None:
        # attn_output shape: [bs, seqlen, num_head/sep, head_dim]
        attn_output = reshard_layer(
            attn_output,
            split_axis=1,
            concat_axis=2,
        )
        # attn_output shape: [bs, seqlen/sep, num_head, head_dim]
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
