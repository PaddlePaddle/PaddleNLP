import paddle 
import numpy as np 
from paddle import Tensor, nn

from custom_setup_ops import neox_rope

dtype = paddle.float32 
batchsize = 1 
numhead = 1
seq_len = 2
head_size = 16

class LlamaRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        dtype = paddle.get_default_dtype()
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, dim, 2), dtype="float32") / dim))
        self.register_buffer("inv_freq", inv_freq.cast(dtype))

        # higher acc using float32
        t = paddle.arange(max_position_embeddings, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq.cast("float32"))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat([freqs, freqs], axis=-1)
        # print("LLAMA ROTARY EMBEDDING IS: ", emb)
        # [bs, seqlen, nhead, head_dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :seq_len, :, ...],
            self.sin_cached[:, :seq_len, :, ...],
        )

def get_rotary_embedding(bsz, max_position_embeddings, base, head_dim, seq_length, offset): 
    inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, head_dim, 2), "float32") / head_dim))
    t = paddle.arange(max_position_embeddings, dtype=inv_freq.dtype)

    # shape: [S, D/2]
    freqs = paddle.einsum("i,j->ij", t, inv_freq)
    # shape: [S, D]
    emb = paddle.concat([freqs, freqs], axis=-1)

    # shape: [1, S, D]
    emb = paddle.unsqueeze(emb, 0)
    # shape: [1, S, 1, D]
    emb = paddle.unsqueeze(emb, 2)
    # shape: [B, S, 1, D]
    emb = paddle.repeat_interleave(emb, bsz, axis=0)
    # print("Embedding is: ", emb)
    cos_emb = paddle.cos(emb)
    sin_emb = paddle.sin(emb)

    stacked_rotary_emb = paddle.concat([cos_emb, sin_emb], axis=0)
    return stacked_rotary_emb[:, offset: seq_length+offset, :, :]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    q = paddle.transpose(q, perm=[0, 2, 1, 3]) # batch, seq, numhead, head_dim
    k = paddle.transpose(k, perm=[0, 2, 1, 3]) # batch, seq, numhead, head_dim
    cos = cos[:, offset : q.shape[1] + offset, :, :]
    sin = sin[:, offset : q.shape[1] + offset, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def baseline(q, k, bsz, max_position_embeddings, base, head_dim, seq_length, offset): 
    rotary_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings)
    cos, sin = rotary_emb(None, seq_len=seq_length)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, offset=offset)
    q_out = paddle.transpose(q_out, perm=[0, 2, 1, 3])
    k_out = paddle.transpose(k_out, perm=[0, 2, 1, 3])
    return q_out, k_out 


q = paddle.cast(paddle.to_tensor(np.random.randn(batchsize, numhead, seq_len, head_size)), dtype)
k = paddle.cast(paddle.to_tensor(np.random.randn(batchsize, numhead, seq_len, head_size)), dtype)
rotary_embedding = paddle.cast(get_rotary_embedding(batchsize, 2048, 10000, head_size, seq_len, offset=0), paddle.float32)
q_out, k_out = neox_rope(q, k, rotary_embedding)
baseline_q_out, baseline_k_out = baseline(q, k, batchsize, 2048, 10000, head_size, seq_len, offset=0)

print("Q is equal?: ", np.allclose(q_out.numpy(), baseline_q_out.numpy(), atol=1e-3, rtol=1e-3))
print("K is equal?: ", np.allclose(k_out.numpy(), baseline_k_out.numpy(), atol=1e-3, rtol=1e-3))
