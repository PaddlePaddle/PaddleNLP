from collections import namedtuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from generation_utils import GenerationMixin


class UnifiedTransformer(nn.Layer, GenerationMixin):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dropout,
                 activation,
                 normalize_before,
                 vocab_size,
                 type_size,
                 max_seq_len,
                 unk_token_id,
                 bos_token_id,
                 eos_token_id,
                 mask_token_id,
                 pad_token_id,
                 is_infer=False):
        super(UnifiedTransformer, self).__init__()

        self.nhead = nhead
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.is_infer = is_infer

        self.word_embedding_layer = nn.Embedding(vocab_size, d_model)
        self.sent_embedding_layer = nn.Embedding(type_size, d_model)
        self.pos_embedding_layer = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_model * 4,
            dropout,
            activation,
            normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers,
                                             encoder_norm)

        self.fc_layer = nn.Linear(d_model, d_model)
        self.norm_layer = nn.LayerNorm(d_model)
        self.logits_bias = paddle.create_parameter(
            [vocab_size], 'float32', is_bias=True)

        self.dropout_layer = nn.Dropout(dropout)
        self.gelu_layer = nn.GELU()
        self.softmax = nn.Softmax()

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                tgt_pos=None,
                use_cache=False,
                cache=None):
        src, src_mask = self.gen_input(input_ids, token_type_ids, position_ids,
                                       attention_mask)

        if use_cache and cache is None:
            cache = self.encoder.gen_cache(src)

        if cache:
            enc_out, cache = self.encoder(src, src_mask, cache)
        else:
            enc_out = self.encoder(src, src_mask)
        logits = self.calc_logits(enc_out, tgt_pos)

        if use_cache:
            return {'logits': logits, 'cache': cache}
        else:
            return {'logits': logits}

    def gen_input(self, token_ids, type_ids, pos_ids, input_mask):
        token_emb_out = self.word_embedding_layer(token_ids)
        type_emb_out = self.sent_embedding_layer(type_ids)
        pos_emb_out = self.pos_embedding_layer(pos_ids)

        emb_out = token_emb_out + type_emb_out + pos_emb_out
        emb_out = self.dropout_layer(emb_out)

        # generate n-head self-attention mask
        self_attn_mask = input_mask
        self_attn_mask = paddle.scale(
            x=self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.stack(
            x=[self_attn_mask] * self.nhead, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        return emb_out, n_head_self_attn_mask

    def calc_logits(self, enc_out, tgt_pos=None):
        if tgt_pos is not None:
            # [batch_size * seq_len, d_model]
            enc_out = paddle.reshape(enc_out, [-1, enc_out.shape[-1]])
            # [x, d_model]
            out = paddle.gather(enc_out, tgt_pos)
        else:
            out = enc_out
        out = self.fc_layer(out)
        out = self.gelu_layer(out)
        out = self.norm_layer(out)
        logits = paddle.matmul(
            out, self.word_embedding_layer.weight,
            transpose_y=True) + self.logits_bias
        return logits

    def logits_preprocess(self, logits):
        # pre-process distribution
        logits[:, self.unk_token_id] = -1e9
        logits[:, self.mask_token_id] = -1e9
        logits[:, self.bos_token_id] = -1e9
        return logits

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        # only last token for inputs_ids if cache is defined in kwargs
        if cache:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1, :].unsqueeze(1)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }
