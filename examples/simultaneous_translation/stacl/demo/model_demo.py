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

import os
import sys

import paddle
import paddle.nn.functional as F

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from model import SimultaneousTransformer


class SimultaneousTransformerDemo(SimultaneousTransformer):
    """
    model
    """

    def greedy_search(self,
                      src_word,
                      max_len=256,
                      waitk=-1,
                      caches=None,
                      bos_id=None):
        """
        greedy_search uses streaming reader. It doesn't need calling
        encoder many times, an a sub-sentence just needs calling encoder once.
        So, it needsprevious state(caches) and last one of generated
        tokens id last time.
        """
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
        enc_outputs = [self.encoder(enc_input, src_mask=src_slf_attn_bias)]

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
        if not bos_id:
            trg_word = paddle.full(shape=[batch_size, 1],
                                   fill_value=self.bos_id,
                                   dtype="int64")
        else:
            trg_word = paddle.full(shape=[batch_size, 1],
                                   fill_value=bos_id,
                                   dtype="int64")

        # init states (caches) for transformer
        if not caches:
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

        return finished_seq, finished_scores, caches
