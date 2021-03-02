import paddle
import paddle.nn as nn
import numpy as np


class StackModel(nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_ids, src_sids, src_pids, tgt_ids, tgt_sids, tgt_pids,
                attn_ids, mask_src_2_src, mask_tgt_2_srctgt,
                mask_attn_2_srctgtattn, tgt_labels, tgt_pos):
        _, __, info = self.model(
            src_ids,
            sent_ids=src_sids,
            pos_ids=src_pids,
            attn_bias=mask_src_2_src,
            encode_only=True)
        cached_k, cached_v = info['caches']
        _, __, info = self.model(
            tgt_ids,
            sent_ids=tgt_sids,
            pos_ids=tgt_pids,
            attn_bias=mask_tgt_2_srctgt,
            past_cache=(cached_k, cached_v),
            encode_only=True)
        cached_k2, cached_v2 = info['caches']
        past_cache_k = [
            paddle.concat([k, k2], 1) for k, k2 in zip(cached_k, cached_k2)
        ]
        past_cache_v = [
            paddle.concat([v, v2], 1) for v, v2 in zip(cached_v, cached_v2)
        ]
        loss, _, __ = self.model(
            attn_ids,
            sent_ids=tgt_sids,
            pos_ids=tgt_pids,
            attn_bias=mask_attn_2_srctgtattn,
            past_cache=(past_cache_k, past_cache_v),
            tgt_labels=tgt_labels,
            tgt_pos=tgt_pos)
        return loss
