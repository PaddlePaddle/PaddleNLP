#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import re
import argparse
import logging
import json
import numpy as np
from collections import namedtuple

import paddle
import paddle.nn as nn
import numpy as np
from paddlenlp.utils.log import logger


def gen_bias(encoder_inputs, decoder_inputs, step):
    decoder_bsz, decoder_seqlen = decoder_inputs.shape[:2]
    encoder_bsz, encoder_seqlen = encoder_inputs.shape[:2]
    attn_bias = paddle.reshape(
        paddle.arange(0, decoder_seqlen, 1, dtype='float32') + 1, [1, -1, 1])
    decoder_bias = paddle.cast(
        (paddle.matmul(attn_bias, 1. / attn_bias, transpose_y=True) >= 1.),
        'float32')  #[1, decoderlen, decoderlen]
    encoder_bias = paddle.unsqueeze(
        paddle.cast(paddle.ones_like(encoder_inputs), 'float32'),
        [1])  #[bsz, 1, encoderlen]
    encoder_bias = paddle.expand(encoder_bias,
                                 [encoder_bsz, decoder_seqlen, encoder_seqlen
                                  ])  #[bsz,decoderlen, encoderlen]
    decoder_bias = paddle.expand(decoder_bias,
                                 [decoder_bsz, decoder_seqlen, decoder_seqlen
                                  ])  #[bsz, decoderlen, decoderlen]
    if step > 0:
        bias = paddle.concat([
            encoder_bias,
            paddle.ones([decoder_bsz, decoder_seqlen, step], 'float32'),
            decoder_bias
        ], -1)
    else:
        bias = paddle.concat([encoder_bias, decoder_bias], -1)
    return bias


@paddle.no_grad()
def greedy_search_infilling(model,
                            token_ids,
                            token_type_ids,
                            sos_id,
                            eos_id,
                            attn_id,
                            pad_id,
                            unk_id,
                            vocab_size,
                            max_encode_len=640,
                            max_decode_len=100,
                            tgt_type_id=3):
    _, logits, info = model(token_ids, token_type_ids)
    d_batch, d_seqlen = token_ids.shape
    seqlen = paddle.sum(paddle.cast(token_ids != 0, 'int64'), 1, keepdim=True)
    has_stopped = np.zeros([d_batch], dtype=np.bool)
    gen_seq_len = np.zeros([d_batch], dtype=np.int64)
    output_ids = []

    past_cache = info['caches']

    cls_ids = paddle.ones([d_batch], dtype='int64') * sos_id
    attn_ids = paddle.ones([d_batch], dtype='int64') * attn_id
    ids = paddle.stack([cls_ids, attn_ids], -1)
    for step in range(max_decode_len):
        bias = gen_bias(token_ids, ids, step)
        pos_ids = paddle.to_tensor(
            np.tile(np.array([[step, step + 1]], dtype=np.int64), [d_batch, 1]))
        pos_ids += seqlen
        _, logits, info = model(ids,
                                paddle.ones_like(ids) * tgt_type_id,
                                pos_ids=pos_ids,
                                attn_bias=bias,
                                past_cache=past_cache)

        if logits.shape[-1] > vocab_size:
            logits[:, :, vocab_size:] = 0
        logits[:, :, pad_id] = 0
        logits[:, :, unk_id] = 0
        logits[:, :, attn_id] = 0

        gen_ids = paddle.argmax(logits, -1)

        past_cached_k, past_cached_v = past_cache
        cached_k, cached_v = info['caches']
        cached_k = [
            paddle.concat([pk, k[:, :1, :]], 1)
            for pk, k in zip(past_cached_k, cached_k)
        ]  # concat cached
        cached_v = [
            paddle.concat([pv, v[:, :1, :]], 1)
            for pv, v in zip(past_cached_v, cached_v)
        ]
        past_cache = (cached_k, cached_v)

        gen_ids = gen_ids[:, 1]
        ids = paddle.stack([gen_ids, attn_ids], 1)

        gen_ids = gen_ids.numpy()
        has_stopped |= (gen_ids == eos_id).astype(np.bool)
        gen_seq_len += (1 - has_stopped.astype(np.int64))
        output_ids.append(gen_ids.tolist())
        if has_stopped.all():
            break
    output_ids = np.array(output_ids).transpose([1, 0])
    return output_ids


BeamSearchState = namedtuple('BeamSearchState',
                             ['log_probs', 'lengths', 'finished'])
BeamSearchOutput = namedtuple('BeamSearchOutput',
                              ['scores', 'predicted_ids', 'beam_parent_ids'])


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


def mask_prob(p, onehot_eos, finished):
    is_finished = paddle.cast(paddle.reshape(finished, [-1, 1]) != 0, 'float32')
    p = is_finished * (1. - paddle.cast(onehot_eos, 'float32')) * -9999. + (
        1. - is_finished) * p
    return p


def hyp_score(log_probs, length, length_penalty):
    lp = paddle.pow((5. + paddle.cast(length, 'float32')) / 6., length_penalty)
    return log_probs / lp


def beam_search_step(state, logits, eos_id, beam_width, is_first_step,
                     length_penalty):
    """logits.shape == [B*W, V]"""
    _, vocab_size = logits.shape

    bsz, beam_width = state.log_probs.shape
    onehot_eos = paddle.cast(
        nn.functional.one_hot(paddle.ones([1], 'int64') * eos_id, vocab_size),
        'int64')  #[1, V]

    probs = paddle.log(nn.functional.softmax(logits))  #[B*W, V]
    probs = mask_prob(probs, onehot_eos, state.finished)  #[B*W, V]
    allprobs = paddle.reshape(state.log_probs, [-1, 1]) + probs  #[B*W, V]

    not_finished = 1 - paddle.reshape(state.finished, [-1, 1])  #[B*W,1]
    not_eos = 1 - onehot_eos
    length_to_add = not_finished * not_eos  #[B*W,V]
    alllen = paddle.reshape(state.lengths, [-1, 1]) + length_to_add

    allprobs = paddle.reshape(allprobs, [-1, beam_width * vocab_size])
    alllen = paddle.reshape(alllen, [-1, beam_width * vocab_size])
    allscore = hyp_score(allprobs, alllen, length_penalty)
    if is_first_step:
        allscore = paddle.reshape(
            allscore,
            [bsz, beam_width, -1])[:, 0, :]  # first step only consiter beam 0
    scores, idx = paddle.topk(allscore, k=beam_width)  #[B, W]
    next_beam_id = idx // vocab_size  #[B, W]
    next_word_id = idx % vocab_size

    gather_idx = paddle.concat(
        [paddle.nonzero(idx != -1)[:, :1],
         paddle.reshape(idx, [-1, 1])], 1)
    next_probs = paddle.reshape(paddle.gather_nd(allprobs, gather_idx),
                                idx.shape)
    next_len = paddle.reshape(paddle.gather_nd(alllen, gather_idx), idx.shape)

    gather_idx = paddle.concat([
        paddle.nonzero(next_beam_id != -1)[:, :1],
        paddle.reshape(next_beam_id, [-1, 1])
    ], 1)
    next_finished = paddle.reshape(
        paddle.gather_nd(state.finished, gather_idx),
        state.finished.shape)  #[gather new beam state according to new beam id]

    next_finished += paddle.cast(next_word_id == eos_id, 'int64')
    next_finished = paddle.cast(next_finished > 0, 'int64')

    next_state = BeamSearchState(log_probs=next_probs,
                                 lengths=next_len,
                                 finished=next_finished)
    output = BeamSearchOutput(scores=scores,
                              predicted_ids=next_word_id,
                              beam_parent_ids=next_beam_id)

    return output, next_state


@paddle.no_grad()
def beam_search_infilling(model,
                          token_ids,
                          token_type_ids,
                          sos_id,
                          eos_id,
                          attn_id,
                          pad_id,
                          unk_id,
                          vocab_size,
                          max_encode_len=640,
                          max_decode_len=100,
                          beam_width=5,
                          tgt_type_id=3,
                          length_penalty=1.0):
    _, __, info = model(token_ids, token_type_ids)
    d_batch, d_seqlen = token_ids.shape

    state = BeamSearchState(log_probs=paddle.zeros([d_batch, beam_width],
                                                   'float32'),
                            lengths=paddle.zeros([d_batch, beam_width],
                                                 'int64'),
                            finished=paddle.zeros([d_batch, beam_width],
                                                  'int64'))
    outputs = []

    def reorder_(t, parent_id):
        """reorder cache according to parent beam id"""
        gather_idx = paddle.nonzero(
            parent_id != -1)[:, 0] * beam_width + paddle.reshape(
                parent_id, [-1])
        t = paddle.gather(t, gather_idx)
        return t

    def tile_(t, times):
        _shapes = list(t.shape[1:])
        new_shape = [t.shape[0], times] + list(t.shape[1:])
        ret = paddle.reshape(paddle.expand(paddle.unsqueeze(t, [1]), new_shape),
                             [
                                 -1,
                             ] + _shapes)
        return ret

    cached_k, cached_v = info['caches']
    cached_k = [tile_(k, beam_width) for k in cached_k]
    cached_v = [tile_(v, beam_width) for v in cached_v]
    past_cache = (cached_k, cached_v)

    token_ids = tile_(token_ids, beam_width)
    seqlen = paddle.sum(paddle.cast(token_ids != 0, 'int64'), 1, keepdim=True)
    #log.debug(token_ids.shape)

    cls_ids = paddle.ones([d_batch * beam_width], dtype='int64') * sos_id
    attn_ids = paddle.ones([d_batch * beam_width],
                           dtype='int64') * attn_id  # SOS
    ids = paddle.stack([cls_ids, attn_ids], -1)
    for step in range(max_decode_len):
        #log.debug('decode step %d' % step)
        bias = gen_bias(token_ids, ids, step)
        pos_ids = paddle.to_tensor(
            np.tile(np.array([[step, step + 1]], dtype=np.int64),
                    [d_batch * beam_width, 1]))
        pos_ids += seqlen
        _, logits, info = model(ids,
                                paddle.ones_like(ids) * tgt_type_id,
                                pos_ids=pos_ids,
                                attn_bias=bias,
                                past_cache=past_cache)
        if logits.shape[-1] > vocab_size:
            logits[:, :, vocab_size:] = 0
        logits[:, :, pad_id] = 0
        logits[:, :, unk_id] = 0
        logits[:, :, attn_id] = 0

        output, state = beam_search_step(state,
                                         logits[:, 1],
                                         eos_id=eos_id,
                                         beam_width=beam_width,
                                         is_first_step=(step == 0),
                                         length_penalty=length_penalty)
        outputs.append(output)

        past_cached_k, past_cached_v = past_cache
        cached_k, cached_v = info['caches']
        cached_k = [
            reorder_(paddle.concat([pk, k[:, :1, :]], 1),
                     output.beam_parent_ids)
            for pk, k in zip(past_cached_k, cached_k)
        ]  # concat cached
        cached_v = [
            reorder_(paddle.concat([pv, v[:, :1, :]], 1),
                     output.beam_parent_ids)
            for pv, v in zip(past_cached_v, cached_v)
        ]
        past_cache = (cached_k, cached_v)

        pred_ids_flatten = paddle.reshape(output.predicted_ids,
                                          [d_batch * beam_width])
        ids = paddle.stack([pred_ids_flatten, attn_ids], 1)

        if state.finished.numpy().all():
            break

    final_ids = paddle.stack([o.predicted_ids for o in outputs], 0)
    final_parent_ids = paddle.stack([o.beam_parent_ids for o in outputs], 0)
    final_ids = nn.functional.gather_tree(final_ids,
                                          final_parent_ids)[:, :,
                                                            0]  #pick best beam
    final_ids = paddle.transpose(paddle.reshape(final_ids, [-1, d_batch * 1]),
                                 [1, 0])

    return final_ids.numpy()


en_patten = re.compile(r'^[a-zA-Z0-9]*$')


def post_process(token):
    if token.startswith('##'):
        ret = token[2:]
    elif token in ['[CLS]', '[SEP]', '[PAD]']:
        ret = ''
    else:
        if en_patten.match(token):
            ret = ' ' + token
        else:
            ret = token
    return ret
