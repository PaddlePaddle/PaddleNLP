# encoding=utf-8
import numpy as np
import paddle

def get_rotary_position_embedding(position_ids, head_dim):
    """
    Pre-calculate rotary position embedding for position_ids.

    Args:
        position_ids: [1, S]
        head_dim: D

    Returns:
        rot_emb: [2, 1, S, 1, D]
    """
    bsz, max_seq_len = position_ids.shape[:2]
    rot_emb = np.zeros((2, bsz, max_seq_len, 1, head_dim // 2), dtype="float32")
    inv_freq = 10000 ** (-np.arange(0, head_dim, 2, dtype="float32") / head_dim)

    # shape: [B, S, D/2]
    freqs = np.einsum("ij,k->ijk", position_ids.numpy().astype("float32"), inv_freq)
    # shape: [B, S, 1, D]
    emb = np.stack([freqs], axis=-1).reshape((bsz, max_seq_len, head_dim // 2))
    emb = np.expand_dims(emb, 2)

    rot_emb[0] = np.cos(emb)
    rot_emb[1] = np.sin(emb)
    rot_emb = np.concatenate([rot_emb, rot_emb], axis=3).transpose([0, 1, 2, 4, 3]).reshape(
        [2, bsz, max_seq_len, 1, head_dim])
    rot_emb = paddle.to_tensor(rot_emb)

    return rot_emb