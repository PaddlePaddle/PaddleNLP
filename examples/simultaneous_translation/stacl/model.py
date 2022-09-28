# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import WordEmbedding, PositionalEmbedding


class CrossEntropyCriterion(nn.Layer):

    def __init__(self, label_smooth_eps, pad_idx=0):
        super(CrossEntropyCriterion, self).__init__()
        self.label_smooth_eps = label_smooth_eps
        self.pad_idx = pad_idx

    def forward(self, predict, label):
        weights = paddle.cast(label != self.pad_idx,
                              dtype=paddle.get_default_dtype())
        if self.label_smooth_eps:
            label = F.label_smooth(label=F.one_hot(
                x=label, num_classes=predict.shape[-1]),
                                   epsilon=self.label_smooth_eps)

        cost = F.cross_entropy(
            input=predict,
            label=label,
            reduction='none',
            soft_label=True if self.label_smooth_eps else False).squeeze()
        weighted_cost = cost * weights
        sum_cost = paddle.sum(weighted_cost)
        token_num = paddle.sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, token_num


class DecoderLayer(nn.TransformerDecoderLayer):

    def __init__(self, *args, **kwargs):
        super(DecoderLayer, self).__init__(*args, **kwargs)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if len(memory) == 1:
            # Full sent
            tgt = self.cross_attn(tgt, memory[0], memory[0], memory_mask, None)
        else:
            # Wait-k policy
            cross_attn_outputs = []
            for i in range(tgt.shape[1]):
                q = tgt[:, i:i + 1, :]
                if i >= len(memory):
                    e = memory[-1]
                else:
                    e = memory[i]
                cross_attn_outputs.append(
                    self.cross_attn(q, e, e, memory_mask[:, :,
                                                         i:i + 1, :e.shape[1]],
                                    None))
            tgt = paddle.concat(cross_attn_outputs, axis=1)
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache, ))


class Decoder(nn.TransformerDecoder):
    """
    PaddlePaddle 2.1 casts memory_mask.dtype to memory.dtype, but in STACL,
    type of memory is list, having no dtype attribute.
    """

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        output = tgt
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             cache=None)
            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)


class SimultaneousTransformer(nn.Layer):
    """
    model
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 weight_sharing,
                 bos_id=0,
                 eos_id=1,
                 waitk=-1):
        super(SimultaneousTransformer, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout
        self.waitk = waitk
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model

        self.src_word_embedding = WordEmbedding(vocab_size=src_vocab_size,
                                                emb_dim=d_model,
                                                bos_id=self.bos_id)
        self.src_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                     max_length=max_length)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            self.trg_word_embedding = self.src_word_embedding
            self.trg_pos_embedding = self.src_pos_embedding
        else:
            self.trg_word_embedding = WordEmbedding(vocab_size=trg_vocab_size,
                                                    emb_dim=d_model,
                                                    bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                         max_length=max_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=d_inner_hid,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   normalize_before=True,
                                                   bias_attr=[False, True])
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=n_layer,
                                             norm=encoder_norm)

        decoder_layer = DecoderLayer(d_model=d_model,
                                     nhead=n_head,
                                     dim_feedforward=d_inner_hid,
                                     dropout=dropout,
                                     activation='relu',
                                     normalize_before=True,
                                     bias_attr=[False, False, True])
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer=decoder_layer,
                               num_layers=n_layer,
                               norm=decoder_norm)

        if weight_sharing:
            self.linear = lambda x: paddle.matmul(x=x,
                                                  y=self.trg_word_embedding.
                                                  word_embedding.weight,
                                                  transpose_y=True)
        else:
            self.linear = nn.Linear(in_features=d_model,
                                    out_features=trg_vocab_size,
                                    bias_attr=False)

    def forward(self, src_word, trg_word):
        src_max_len = paddle.shape(src_word)[-1]
        trg_max_len = paddle.shape(trg_word)[-1]
        base_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        src_slf_attn_bias = base_attn_bias
        src_slf_attn_bias.stop_gradient = True
        trg_slf_attn_bias = paddle.tensor.triu((paddle.ones(
            (trg_max_len, trg_max_len), dtype=paddle.get_default_dtype()) *
                                                -np.inf), 1)
        trg_slf_attn_bias.stop_gradient = True
        trg_src_attn_bias = paddle.tile(base_attn_bias, [1, 1, trg_max_len, 1])
        src_pos = paddle.cast(src_word != self.bos_id,
                              dtype="int64") * paddle.arange(start=0,
                                                             end=src_max_len)
        trg_pos = paddle.cast(trg_word != self.bos_id,
                              dtype="int64") * paddle.arange(start=0,
                                                             end=trg_max_len)
        src_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=self.training) if self.dropout else src_emb
        with paddle.static.amp.fp16_guard():
            if self.waitk >= src_max_len or self.waitk == -1:
                # Full sentence
                enc_outputs = [
                    self.encoder(enc_input, src_mask=src_slf_attn_bias)
                ]
            else:
                # Wait-k policy
                enc_outputs = []
                for i in range(self.waitk, src_max_len + 1):
                    enc_output = self.encoder(
                        enc_input[:, :i, :],
                        src_mask=src_slf_attn_bias[:, :, :, :i])
                    enc_outputs.append(enc_output)

            trg_emb = self.trg_word_embedding(trg_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb
            dec_output = self.decoder(dec_input,
                                      enc_outputs,
                                      tgt_mask=trg_slf_attn_bias,
                                      memory_mask=trg_src_attn_bias)

            predict = self.linear(dec_output)

        return predict

    def beam_search(self, src_word, beam_size=4, max_len=256, waitk=-1):
        # TODO: "Speculative Beam Search for Simultaneous Translation"
        raise NotImplementedError

    def greedy_search(self, src_word, max_len=256, waitk=-1):
        src_max_len = paddle.shape(src_word)[-1]
        base_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        src_slf_attn_bias = base_attn_bias
        src_slf_attn_bias.stop_gradient = True
        trg_src_attn_bias = paddle.tile(base_attn_bias, [1, 1, 1, 1])
        src_pos = paddle.cast(src_word != self.bos_id,
                              dtype="int64") * paddle.arange(start=0,
                                                             end=src_max_len)
        src_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=self.training) if self.dropout else src_emb
        if waitk < 0 or waitk > src_max_len:
            enc_outputs = [self.encoder(enc_input, src_mask=src_slf_attn_bias)]
        else:
            enc_outputs = []
            for i in range(waitk, src_max_len + 1):
                enc_output = self.encoder(
                    enc_input[:, :i, :],
                    src_mask=src_slf_attn_bias[:, :, :, :i])
                enc_outputs.append(enc_output)

        # constant number
        batch_size = enc_outputs[-1].shape[0]
        max_len = (enc_outputs[-1].shape[1] +
                   20) if max_len is None else max_len
        end_token_tensor = paddle.full(shape=[batch_size, 1],
                                       fill_value=self.eos_id,
                                       dtype="int64")

        predict_ids = []
        log_probs = paddle.full(shape=[batch_size, 1],
                                fill_value=0,
                                dtype="float32")
        trg_word = paddle.full(shape=[batch_size, 1],
                               fill_value=self.bos_id,
                               dtype="int64")

        # init states (caches) for transformer
        caches = self.decoder.gen_cache(enc_outputs[-1], do_zip=False)

        for i in range(max_len):
            trg_pos = paddle.full(shape=trg_word.shape,
                                  fill_value=i,
                                  dtype="int64")
            trg_emb = self.trg_word_embedding(trg_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb

            if waitk < 0 or i >= len(enc_outputs):
                # Avoid getting the whole source in advance, a diff from:
                # https://github.com/autosimtrans/SimulTransBaseline/blob/master/model.py#L1207
                # if the decoder step is full sent or longer than all source
                # step, then read the whole src
                _e = enc_outputs[-1]
                dec_output, caches = self.decoder(
                    dec_input, [_e], None,
                    trg_src_attn_bias[:, :, :, :_e.shape[1]], caches)
            else:
                _e = enc_outputs[i]
                dec_output, caches = self.decoder(
                    dec_input, [_e], None,
                    trg_src_attn_bias[:, :, :, :_e.shape[1]], caches)

            dec_output = paddle.reshape(dec_output,
                                        shape=[-1, dec_output.shape[-1]])

            logits = self.linear(dec_output)
            step_log_probs = paddle.log(F.softmax(logits, axis=-1))
            log_probs = paddle.add(x=step_log_probs, y=log_probs)
            scores = log_probs
            topk_scores, topk_indices = paddle.topk(x=scores, k=1)

            finished = paddle.equal(topk_indices, end_token_tensor)
            trg_word = topk_indices
            log_probs = topk_scores

            predict_ids.append(topk_indices)

            if paddle.all(finished).numpy():
                break

        predict_ids = paddle.stack(predict_ids, axis=0)
        finished_seq = paddle.transpose(predict_ids, [1, 2, 0])
        finished_scores = topk_scores

        return finished_seq, finished_scores
