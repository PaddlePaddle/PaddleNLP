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
import sys
import os
import numpy as np
from attrdict import AttrDict
import argparse
import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import yaml
from pprint import pprint

from paddlenlp.ops import InferTransformerDecoder
from paddlenlp.utils.log import logger

from paddlenlp.transformers import WordEmbedding, PositionalEmbedding, position_encoding_init


class FasterDecoder(nn.Layer):
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
                 max_out_len=256,
                 decoder_lib=None,
                 use_fp16_decoder=False):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout
        self.max_out_len = max_out_len
        self.max_length = max_length
        self.use_fp16_decoder = use_fp16_decoder
        self.n_layer = n_layer
        self.d_model = d_model

        self.src_word_embedding = WordEmbedding(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
        # print(self.src_word_embedding.word_embedding.weight)
        self.src_pos_embedding = PositionalEmbedding(
            emb_dim=d_model, max_length=max_length)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            self.trg_word_embedding = self.src_word_embedding
            self.trg_pos_embedding = self.src_pos_embedding
        else:
            self.trg_word_embedding = WordEmbedding(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(
                emb_dim=d_model, max_length=max_length)

        self.transformer = paddle.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            activation="relu",
            normalize_before=True)

        self.decoder = InferTransformerDecoder(
            decoder=self.transformer.decoder,
            n_head=n_head,
            size_per_head=d_model // n_head,
            decoder_lib=decoder_lib,
            use_fp16_decoder=use_fp16_decoder)

        if weight_sharing:
            self.linear = lambda x: paddle.matmul(x=x,
                                                  y=self.trg_word_embedding.word_embedding.weight,
                                                  transpose_y=True)
        else:
            self.linear = nn.Linear(
                in_features=d_model,
                out_features=trg_vocab_size,
                bias_attr=False)

    def forward(self, src_word):
        batch_size, src_max_len = paddle.shape(src_word)
        mem_seq_lens = paddle.full(
            shape=[batch_size, 1], fill_value=src_max_len, dtype='int32')

        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9

        src_slf_attn_bias.stop_gradient = True

        src_pos = paddle.cast(
            src_word != self.bos_id, dtype="int64") * paddle.arange(
                start=0, end=src_max_len)

        src_emb = self.src_word_embedding(src_word)

        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=self.training) if self.dropout else src_emb
        enc_output = self.transformer.encoder(
            enc_input, src_mask=src_slf_attn_bias)

        batch_size = enc_output.shape[0]
        end_token_tensor = paddle.full(
            shape=[batch_size, 1], fill_value=self.eos_id, dtype="int64")

        predict_ids = []
        log_probs = paddle.full(
            shape=[batch_size, 1], fill_value=0, dtype="float32")
        trg_word = paddle.full(
            shape=[batch_size, 1], fill_value=self.bos_id, dtype="int64")

        if args.use_fp16_decoder:
            enc_output = paddle.cast(enc_output, "float16")

        # Init cache
        self_cache = paddle.zeros(
            shape=[self.n_layer, 2, 0, batch_size, self.d_model],
            dtype=enc_output.dtype)
        mem_cache = paddle.zeros(
            shape=[self.n_layer, 2, batch_size, src_max_len, self.d_model],
            dtype=enc_output.dtype)

        for i in range(args.max_out_len):
            trg_pos = paddle.full(
                shape=trg_word.shape, fill_value=i, dtype="int64")
            trg_emb = self.trg_word_embedding(trg_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb

            if args.use_fp16_decoder:
                dec_input = paddle.cast(dec_input, "float16")

            dec_output, self_cache, mem_cache = self.decoder(
                from_tensor=dec_input,
                memory_tensor=enc_output,
                mem_seq_len=mem_seq_lens,
                self_cache=self_cache,
                mem_cache=mem_cache)

            if args.use_fp16_decoder:
                dec_output = paddle.cast(dec_output, "float32")

            dec_output = paddle.reshape(
                dec_output, shape=[-1, dec_output.shape[-1]])

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

    def load(self, init_from_params):
        # Load the trained model
        assert init_from_params, (
            "Please set init_from_params to load the infer model.")

        model_dict = paddle.load(init_from_params, return_numpy=True)

        # To set weight[padding_idx] to 0.
        model_dict["trg_word_embedding.word_embedding.weight"][
            self.bos_id] = [0] * self.d_model

        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)
        model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)

        if self.use_fp16_decoder:
            for item in self.state_dict():
                if "decoder.layers" in item:
                    model_dict[item] = np.float16(model_dict[item])

        self.load_dict(model_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./config/decoder.sample.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--decoder_lib",
        default="../../build/lib/libdecoder_op.so",
        type=str,
        help="Path of libdecoder_op.so. ")
    parser.add_argument(
        "--use_fp16_decoder",
        action="store_true",
        help="Whether to use fp16 decoder to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    paddle.set_device(place)
    paddle.seed(5678)

    # Define model
    transformer = FasterDecoder(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        max_out_len=args.max_out_len,
        decoder_lib=args.decoder_lib,
        use_fp16_decoder=args.use_fp16_decoder)

    # Load checkpoint.
    transformer.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))
    # Set evaluate mode
    transformer.eval()

    src_word = paddle.randint(
        0, 30000, shape=[args.infer_batch_size, args.max_length], dtype='int64')

    with paddle.no_grad():
        for i in range(100):
            # For warmup. 
            if 50 == i:
                start = time.time()
            finished_seq, finished_scores = transformer(src_word=src_word)
        logger.info("Average test time for decoder is %f ms" % (
            (time.time() - start) / 50 * 1000))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.decoder_lib = ARGS.decoder_lib
    args.use_fp16_decoder = ARGS.use_fp16_decoder

    do_predict(args)
