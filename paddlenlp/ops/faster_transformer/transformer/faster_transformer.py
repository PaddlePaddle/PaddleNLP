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
import shutil
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import (TransformerModel, WordEmbedding,
                                    PositionalEmbedding, position_encoding_init,
                                    InferTransformerModel, GPTModel)
from paddlenlp.ops import InferTransformerDecoding, InferGptDecoding
from paddlenlp.ops.ext_utils import load
from paddlenlp.utils.log import logger
from paddlenlp.transformers import GPTChineseTokenizer, GPTTokenizer


class FasterTransformer(TransformerModel):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 weight_sharing,
                 attn_dropout=None,
                 act_dropout=None,
                 bos_id=0,
                 eos_id=1,
                 decoding_strategy="beam_search",
                 beam_size=4,
                 topk=1,
                 topp=0.0,
                 max_out_len=256,
                 decoding_lib=None,
                 use_fp16_decoding=False):
        # if decoding_lib is None:
        #     raise ValueError(
        #         "The args decoding_lib must be set to use Faster Transformer. ")
        # elif not os.path.exists(decoding_lib):
        #     raise ValueError("The path to decoding lib is not exist.")

        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.decoding_strategy = args.pop("decoding_strategy")
        self.beam_size = args.pop("beam_size")
        self.topk = args.pop("topk")
        self.topp = args.pop("topp")
        self.max_out_len = args.pop("max_out_len")
        self.decoding_lib = args.pop("decoding_lib")
        self.use_fp16_decoding = args.pop("use_fp16_decoding")
        self.dropout = dropout
        self.weight_sharing = weight_sharing
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.bos_id = bos_id
        self.max_length = max_length
        super(FasterTransformer, self).__init__(**args)

        self.decoding_linear = nn.Linear(
            in_features=d_model, out_features=trg_vocab_size)

        if weight_sharing:
            self.trg_word_embedding = WordEmbedding(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(
                emb_dim=d_model, max_length=max_length)

        self.decoding = InferTransformerDecoding(
            decoder=self.transformer.decoder,
            word_embedding=self.trg_word_embedding.word_embedding,
            positional_embedding=self.trg_pos_embedding.pos_encoder,
            linear=self.decoding_linear,
            num_decoder_layers=num_decoder_layers,
            n_head=n_head,
            d_model=d_model,
            bos_id=bos_id,
            eos_id=eos_id,
            decoding_strategy=decoding_strategy,
            beam_size=beam_size,
            topk=topk,
            topp=topp,
            max_out_len=max_out_len,
            decoding_lib=self.decoding_lib,
            use_fp16_decoding=self.use_fp16_decoding)

    def forward(self, src_word):
        src_max_len = paddle.shape(src_word)[-1]
        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        src_pos = paddle.cast(
            src_word != self.bos_id, dtype=src_word.dtype) * paddle.arange(
                start=0, end=src_max_len)

        # Run encoder
        src_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=False) if self.dropout else src_emb
        enc_output = self.transformer.encoder(enc_input, src_slf_attn_bias)

        if self.use_fp16_decoding:
            enc_output = paddle.cast(enc_output, dtype="float16")

        mem_seq_lens = paddle.sum(paddle.cast(
            src_word != self.bos_id, dtype="int32"),
                                  dtype="int32",
                                  axis=1)
        ids = self.decoding(enc_output, mem_seq_lens)

        return ids

    def load(self, init_from_params):
        # Load the trained model
        assert init_from_params, (
            "Please set init_from_params to load the infer model.")

        model_dict = paddle.load(init_from_params, return_numpy=True)

        # To set weight[padding_idx] to 0.
        model_dict["trg_word_embedding.word_embedding.weight"][
            self.bos_id] = [0] * self.d_model

        # Dealing with weight sharing.
        if self.weight_sharing:
            model_dict["decoding_linear.weight"] = np.transpose(model_dict[
                "trg_word_embedding.word_embedding.weight"])
        else:
            model_dict["decoding_linear.weight"] = model_dict["linear.weight"]
        # NOTE: the data type of the embedding bias for logits is different
        # between decoding with beam search and top-k/top-p sampling in
        # Faster Transformer when using float16.
        bias_dtype = "float32"
        if self.use_fp16_decoding and "beam_search" != self.decoding_strategy:
            bias_dtype = "float16"
        model_dict["decoding_linear.bias"] = np.zeros(
            [self.trg_vocab_size], dtype=bias_dtype)

        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)
        model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)

        if self.use_fp16_decoding:
            for item in self.state_dict():
                if "decoder" in item:
                    model_dict[item] = np.float16(model_dict[item])
            model_dict["decoding_linear.weight"] = np.float16(model_dict[
                "decoding_linear.weight"])
            model_dict["trg_word_embedding.word_embedding.weight"] = np.float16(
                model_dict["trg_word_embedding.word_embedding.weight"])
            model_dict["trg_pos_embedding.pos_encoder.weight"] = np.float16(
                model_dict["trg_pos_embedding.pos_encoder.weight"])

        self.load_dict(model_dict)

    def export_params(self, init_from_params, place):
        # Load the trained model
        assert init_from_params, (
            "Please set init_from_params to load the infer model.")

        model_dict = paddle.load(init_from_params, return_numpy=True)

        # To set weight[padding_idx] to 0.
        model_dict["trg_word_embedding.word_embedding.weight"][
            self.bos_id] = [0] * self.d_model

        # Dealing with weight sharing.
        if self.weight_sharing:
            model_dict["decoding_linear.weight"] = np.transpose(model_dict[
                "trg_word_embedding.word_embedding.weight"])
        else:
            model_dict["decoding_linear.weight"] = model_dict["linear.weight"]
        # NOTE: the data type of the embedding bias for logits is different
        # between decoding with beam search and top-k/top-p sampling in
        # Faster Transformer when using float16.
        bias_dtype = "float32"
        if self.use_fp16_decoding and "beam_search" != self.decoding_strategy:
            bias_dtype = "float16"
        model_dict["decoding_linear.bias"] = np.zeros(
            [self.trg_vocab_size], dtype=bias_dtype)

        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)
        model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)

        if self.use_fp16_decoding:
            for item in self.state_dict():
                if "decoder" in item:
                    model_dict[item] = np.float16(model_dict[item])
            model_dict["decoding_linear.weight"] = np.float16(model_dict[
                "decoding_linear.weight"])
            model_dict["trg_word_embedding.word_embedding.weight"] = np.float16(
                model_dict["trg_word_embedding.word_embedding.weight"])
            model_dict["trg_pos_embedding.pos_encoder.weight"] = np.float16(
                model_dict["trg_pos_embedding.pos_encoder.weight"])

        for item in self.state_dict():
            param = self
            attr_list = item.split(".")
            for attr in attr_list:
                param = getattr(param, attr)
            param_name = param.name
            var = paddle.static.global_scope().find_var(param_name).get_tensor()
            var.set(model_dict[item], place)


class TransformerGenerator(paddle.nn.Layer):
    """
    The Transformer model for auto-regressive generation. It wraps `FasterTransformer`
    and `InferTransformerModel`, and automatically chioces using `FasterTransformer`
    (with jit building) or the slower verison `InferTransformerModel`.

    Args:
        src_vocab_size (int):
            The size of source vocabulary.
        trg_vocab_size (int):
            The size of target vocabulary.
        max_length (int):
            The maximum length of input sequences.
        num_encoder_layers (int):
            The number of sub-layers to be stacked in the encoder.
        num_decoder_layers (int):
            The number of sub-layers to be stacked in the decoder.
        n_head (int):
            The number of head used in multi-head attention.
        d_model (int):
            The dimension for word embeddings, which is also the last dimension of
            the input and output of multi-head attention, position-wise feed-forward
            networks, encoder and decoder.
        d_inner_hid (int):
            Size of the hidden layer in position-wise feed-forward networks.
        dropout (float):
            Dropout rates. Used for pre-process, activation and inside attention.
        weight_sharing (bool):
            Whether to use weight sharing. 
        bos_id (int, optional):
            The start token id and also is used as padding id. Defaults to 0.
        eos_id (int, optional):
            The end token id. Defaults to 1.
        beam_size (int, optional):
            The beam width for beam search. Defaults to 4. 
        max_out_len (int, optional):
            The maximum output length. Defaults to 256.
        kwargs:
            The key word arguments can be `output_time_major`, `use_fp16_decoding` and `use_ft`.
            `output_time_major(bool, optional)`: Indicate the data layout of predicted
            Tensor. If `False`, the data layout would be batch major with shape
            `[batch_size, seq_len, beam_size]`. If  `True`, the data layout would
            be time major with shape `[seq_len, batch_size, beam_size]`. Default
            to `False`. `use_fp16_decoding(bool, optional)`: Whether to use fp16
            for decoding. `use_ft(bool, optional)`: Whether to use Faster Transformer
            for decoding. 
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 weight_sharing,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256,
                 **kwargs):
        logger.warning(
            "TransformerGenerator is an experimental API and subject to change.")
        # `kwargs` can include output_time_major, use_fp16_decoding, topk, topp.
        # The later three arguments can only work when using FasterTransformer,
        # and expose topk, topp later.
        super(TransformerGenerator, self).__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.output_time_major = kwargs.pop("output_time_major", True)
        use_fp16_decoding = kwargs.pop("use_fp16_decoding", False)
        use_ft = kwargs.pop("use_ft", True)

        if use_ft:
            try:
                load("FasterTransformer", verbose=True)
                self.transformer = FasterTransformer(
                    src_vocab_size=src_vocab_size,
                    trg_vocab_size=trg_vocab_size,
                    max_length=max_length,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    n_head=n_head,
                    d_model=d_model,
                    d_inner_hid=d_inner_hid,
                    dropout=dropout,
                    weight_sharing=weight_sharing,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    beam_size=beam_size,
                    max_out_len=max_out_len,
                    use_fp16_decoding=use_fp16_decoding)
            except Exception:
                logger.warning(
                    "Exception occurs when using Faster Transformer. " \
                    "The original forward will be involved. ")
                self.transformer = InferTransformerModel(
                    src_vocab_size=src_vocab_size,
                    trg_vocab_size=trg_vocab_size,
                    max_length=max_length,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    n_head=n_head,
                    d_model=d_model,
                    d_inner_hid=d_inner_hid,
                    dropout=dropout,
                    weight_sharing=weight_sharing,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    beam_size=beam_size,
                    max_out_len=max_out_len,
                    output_time_major=self.output_time_major,
                    **kwargs)
        else:
            self.transformer = InferTransformerModel(
                src_vocab_size=src_vocab_size,
                trg_vocab_size=trg_vocab_size,
                max_length=max_length,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                n_head=n_head,
                d_model=d_model,
                d_inner_hid=d_inner_hid,
                dropout=dropout,
                weight_sharing=weight_sharing,
                bos_id=bos_id,
                eos_id=eos_id,
                beam_size=beam_size,
                max_out_len=max_out_len,
                output_time_major=self.output_time_major,
                **kwargs)

    def forward(self, src_word):
        r"""
        Performs decoding for transformer model.

        Args:
            src_word (Tensor):
                The ids of source sequence words. It is a tensor with shape
                `[batch_size, source_sequence_length]` and its data type can be
                int or int64.
        
        Returns:
            Tensor:
                An int64 tensor shaped indicating the predicted ids. Its shape is
                `[batch_size, seq_len, beam_size]` or `[seq_len, batch_size, beam_size]`
                according to `output_time_major`.
        
        Example:
            .. code-block::

                import paddle
                from paddlenlp.ops import TransformerGenerator

                transformer = TransformerGenerator(
                    src_vocab_size=30000,
                    trg_vocab_size=30000,
                    max_length=256,
                    num_encoder_layers=6,
                    num_decoder_layers=6,
                    n_head=8,
                    d_model=512,
                    d_inner_hid=2048,
                    dropout=0.1,
                    weight_sharing=True,
                    bos_id=0,
                    eos_id=1,
                    beam_size=4,
                    max_out_len=256)

                batch_size = 5
                seq_len = 10
                transformer(
                    src_word=paddle.randint(low=3, high=30000, shape=[batch_size, seq_len]))
        """
        out = self.transformer(src_word)
        # TODO(guosheng): FasterTransformer has an output with layout
        # `[seq_len, batch_size, beam_size]`. While the output layout of
        # original one is `[batch_size, seq_len, beam_size]`. Maybe we need
        # unify them later.
        if not self.output_time_major and isinstance(self.transformer,
                                                     FasterTransformer):
            out = paddle.transpose(out, [1, 0, 2])
        return out

    def load(self, path):
        if isinstance(self.transformer, FasterTransformer):
            self.transformer.load(path)
        else:
            model_dict = paddle.load(path)
            self.transformer.load_dict(model_dict)


class FasterGPT(nn.Layer):
    def __init__(self,
                 model,
                 topk=4,
                 topp=0.0,
                 max_out_len=256,
                 bos_id=50256,
                 eos_id=50256,
                 temperature=0,
                 decoding_lib=None,
                 use_fp16_decoding=False):
        super(FasterGPT, self).__init__()
        self.use_fp16_decoding = use_fp16_decoding
        self.decoding = InferGptDecoding(
            model=model,
            topk=topk,
            topp=topp,
            max_out_len=max_out_len,
            bos_id=bos_id,
            eos_id=eos_id,
            temperature=temperature,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding)

    def forward(self, input_ids):
        return self.decoding(input_ids)

    def export_params(self, state_to_load, place):
        for item in state_to_load:
            param_data = np.array(state_to_load[item])
            if self.use_fp16_decoding:
                param_data = np.float16(param_data)

            param = self
            attr_list = item.split(".")
            attr_list = ["decoding", "model"] + attr_list
            for attr in attr_list:
                param = getattr(param, attr)
            param_name = param.name
            var = paddle.static.global_scope().find_var(param_name).get_tensor()
            var.set(param_data, place)

    def save_resources(self, tokenizer, path):
        vocab_file = os.path.join(path, "vocab.txt")
        if isinstance(tokenizer, GPTTokenizer):
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for token in tokenizer.encoder:
                    f.write(token + '\n')
            merges_file = os.path.join(path, "merges.txt")
            shutil.copyfile(tokenizer._merges_file, merges_file)
        elif isinstance(tokenizer, GPTChineseTokenizer):
            tokenizer.save_resources(path)
