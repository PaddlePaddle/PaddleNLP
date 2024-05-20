import paddlenlp
from typing import Optional, Tuple
import math
import warnings
from paddlenlp.transformers.llama.modeling import apply_rotary_pos_emb, repeat_kv, rotate_half
import paddle
import paddle.nn.functional as F
group_size_ratio = 1/4

def ssa_forward(
    self,
    hidden_states,
    position_ids: Optional[Tuple[paddle.Tensor]] = None,
    past_key_value: Optional[Tuple[paddle.Tensor]] = None,
    attention_mask: Optional[paddle.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    alibi: Optional[paddle.Tensor] = None,
):
    """Input shape: Batch x Time x Channel"""
    # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)
    bsz, q_len, _ = hidden_states.shape
    group_size = int(q_len * group_size_ratio)

    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)



    # query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
    # key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
    # value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])

    # 先不transpose，之后补上
    target_query_shape = [0, 0, self.num_heads, self.head_dim]
    target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
    query_states = query_states.reshape(shape=target_query_shape)
    key_states = key_states.reshape(shape=target_key_value_shape)
    value_states = value_states.reshape(shape=target_key_value_shape)

    kv_seq_len = key_states.shape[-3]

    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-3]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # shape = [4, 32, 8192, 128]
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = paddle.concat([past_key_value[0], key_states], axis=1)
        value_states = paddle.concat([past_key_value[1], value_states], axis=1)

    pask_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # transpose   
    # (bsz, q_len, self.num_heads, self.head_dim) -> (bsz, self.num_heads, q_len, self.head_dim)
    query_states = query_states.transpose([0, 2, 1, 3])
    key_states = key_states.transpose([0, 2, 1, 3])
    value_states = value_states.transpose([0, 2, 1, 3])

    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        qkv[:, num_heads // 2:] = qkv[:, num_heads // 2].roll(shifts=(-group_size // 2), axis=2)
        qkv = qkv.transpose([0, 2, 1, 3]).reshape([bsz * (q_len // group_size), group_size, num_heads, head_dim]).transpose([0,2,1,3])
        return qkv

    query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, self.num_key_value_heads, self.head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, self.num_key_value_heads, self.head_dim)

    attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)
    if attn_weights.shape != [bsz * num_group, self.num_heads, group_size, group_size]:
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
            f" {attn_weights.shape}"
        )
    attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)

    if attention_mask is not None:
        if attention_mask.shape != [bsz * num_group, 1, group_size, group_size]:
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.shape}"
            )
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
    attn_output = paddle.matmul(attn_weights, value_states)
    if attn_output.shape != [bse * num_group, self.num_heads, group_size, self.head_dim]:
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.transpose([0, 2, 1, 3])

    attn_output = attn_output.reshape([bsz, q_len, self.num_heads, self.head_dim])

    # shift back
    attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].rool(shift=group_size//2, axis=1)

    attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    output = (attn_output,)
    if output_attentions:
        output += (attn_weights,)
    if use_cache:
        output += (pask_key_value,)
    if type(outputs) is tuple and len(outputs) == 1:
        outputs = outputs[0]
    return outputs

def replace_llama_attn():
    paddlenlp.transformers.llama.modeling.LlamaAttention.forward = ssa_forward

