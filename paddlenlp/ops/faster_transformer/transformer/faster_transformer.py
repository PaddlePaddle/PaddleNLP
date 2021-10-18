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
from paddlenlp.ops import (InferTransformerDecoding, InferGptDecoding,
                           InferUnifiedDecoding, InferBartDecoding)
from paddlenlp.ops.ext_utils import load
from paddlenlp.utils.log import logger
from paddlenlp.transformers import (GPTChineseTokenizer, GPTTokenizer,
                                    UnifiedTransformerPretrainedModel,
                                    UNIMOPretrainedModel, BartPretrainedModel)


class FasterTransformer(TransformerModel):
    """
    FasterTransformer is a faster version for generation with the Transformer
    model. It uses a custom op based on and enhancing NV FasterTransformer to
    do fast generation.

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
        attn_dropout (float):
            The dropout probability used in MHA to drop some attention target.
            If None, use the value of dropout. Defaults to None.
        act_dropout (float):
            The dropout probability used after FFN activition. If None, use
            the value of dropout. Defaults to None.
        bos_id (int, optional):
            The start token id and also is used as padding id. Defaults to 0.
        eos_id (int, optional):
            The end token id. Defaults to 1.
        decoding_strategy (str, optional):
            Indicating the strategy of decoding. It can be 'beam_search', 'beam_search_v2',
            'topk_sampling' and 'topp_sampling'. For beam search strategies,
            'v2' would select the top `beam_size * 2` beams and process the top
            `beam_size` alive and finish beams in them separately, while 'v1'
            would only select the top `beam_size` beams and mix up the alive and
            finish beams. 'v2' always searchs more and get better results, since
            the alive beams would always be `beam_size` while the number of alive
            beams in `v1` might decrease when meeting the end token. However,
            'v2' always generates longer results thus might do more calculation
            and be slower.
        beam_size (int, optional):
            The beam width for beam search. Defaults to 4. 
        topk (int, optional):
            The number of highest probability tokens to keep for top-k sampling.
            Defaults to 4. 
        topp (float, optional):
            The most probable tokens whose cumulative probability is not less than
            `topp` are kept for top-p sampling. Defaults to 4. 
        max_out_len (int, optional):
            The maximum output length. Defaults to 256.
        diversity_rate (float, optional):
            Refer to `A Simple, Fast Diverse Decoding Algorithm for Neural Generation <https://arxiv.org/abs/1611.08562>`_
            for details. Bigger `diversity_rate` would lead to more diversity.
            if `diversity_rate == 0` is equivalent to naive BeamSearch. Default
            to 0 if not set.
        use_fp16_decoding(bool, optional): Whether to use fp16 for decoding. 
        rel_len(bool, optional):
            Indicating whether `max_out_len` in is the length relative to that
            of source text. Only works in `v2` temporarily. It is suggest to set
            a small `max_out_len` and use `rel_len=True`. Default to False if
            not set.
        alpha(float, optional):
            The power number in length penalty calculation. Only works in `v2`
            temporarily. Refer to `GNMT <https://arxiv.org/pdf/1609.08144.pdf>`_.
            Default to 0.6 if not set.
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
                 attn_dropout=None,
                 act_dropout=None,
                 bos_id=0,
                 eos_id=1,
                 decoding_strategy="beam_search",
                 beam_size=4,
                 topk=1,
                 topp=0.0,
                 max_out_len=256,
                 diversity_rate=0.0,
                 decoding_lib=None,
                 use_fp16_decoding=False,
                 rel_len=False,
                 alpha=0.6):
        # if decoding_lib is None:
        #     raise ValueError(
        #         "The args decoding_lib must be set to use FasterTransformer. ")
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
        self.diversity_rate = args.pop("diversity_rate")
        self.decoding_lib = args.pop("decoding_lib")
        self.use_fp16_decoding = args.pop("use_fp16_decoding")
        self.rel_len = args.pop("rel_len")
        self.alpha = args.pop("alpha")
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
            diversity_rate=self.diversity_rate,
            decoding_lib=self.decoding_lib,
            use_fp16_decoding=self.use_fp16_decoding,
            rel_len=self.rel_len,
            alpha=self.alpha)

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
        # FasterTransformer when using float16.
        # NOTE: This changes since FasterTransformer V4.0 and update accordingly
        # after update to FT-4.0.
        bias_dtype = "float32"
        if self.use_fp16_decoding and not self.decoding_strategy.startswith(
                "beam_search"):
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
        '''
        This method is used for load static graph from dygraph checkpoint
        or export inference model using static graph. 

        Args:
            init_from_params (string):
                The path to dygraph checkpoint. 
            place (paddle.Place):
                The place to execute static graph. 
        
        Example:
            .. code-block::
                paddle.enable_static()
                place = "gpu"
                place = paddle.set_device(place)
                reader.adapt_vocab_size(args)

                test_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                with paddle.static.program_guard(test_program, startup_program):
                    src_word = paddle.static.data(
                        name="src_word", shape=[None, None], dtype="int64")

                    # Define model
                    transformer = FasterTransformer(
                        src_vocab_size=args.src_vocab_size,
                        trg_vocab_size=args.trg_vocab_size,
                        max_length=args.max_length + 1,
                        num_encoder_layers=args.n_layer,
                        num_decoder_layers=args.n_layer,
                        n_head=args.n_head,
                        d_model=args.d_model,
                        d_inner_hid=args.d_inner_hid,
                        dropout=args.dropout,
                        weight_sharing=args.weight_sharing,
                        bos_id=args.bos_idx,
                        eos_id=args.eos_idx,
                        decoding_strategy=args.decoding_strategy,
                        beam_size=args.beam_size,
                        max_out_len=args.max_out_len,
                        decoding_lib=args.decoding_lib,
                        use_fp16_decoding=args.use_fp16_decoding,
                        rel_len=args.use_rel_len,
                        alpha=args.alpha)

                    finished_seq = transformer(src_word=src_word)

                test_program = test_program.clone(for_test=True)

                exe = paddle.static.Executor(place)
                exe.run(startup_program)

                # Load checkpoint.
                transformer.export_params(
                    init_from_params=os.path.join(args.init_from_params,
                                                "transformer.pdparams"),
                    place=place)

                paddle.static.save_inference_model(
                    os.path.join(args.inference_model_dir, "transformer"),
                    feed_vars=src_word,
                    fetch_vars=finished_seq,
                    executor=exe,
                    program=test_program)
        '''
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
        # FasterTransformer when using float16.
        # NOTE: This changes since FasterTransformer V4.0 and update accordingly
        # after update to FT-4.0.
        bias_dtype = "float32"
        if self.use_fp16_decoding and not self.decoding_strategy.startswith(
                "beam_search"):
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
    The Transformer model for auto-regressive generation with beam search. It wraps
    `FasterTransformer` and `InferTransformerModel`, and automatically chioces using
    `FasterTransformer` (with jit building) or the slower verison `InferTransformerModel`.

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
            The key word arguments can be `output_time_major`, `use_ft`, `use_fp16_decoding`,
            `rel_len`, `alpha`:

            - `output_time_major(bool, optional)`: Indicate the data layout of predicted
            Tensor. If `False`, the data layout would be batch major with shape
            `[batch_size, seq_len, beam_size]`. If  `True`, the data layout would
            be time major with shape `[seq_len, batch_size, beam_size]`. Default
            to `False`. 

            - `use_ft(bool, optional)`: Whether to use FasterTransformer
            for decoding. Default to True if not set.

            - `use_fp16_decoding(bool, optional)`: Whether to use fp16
            for decoding.  Only works when using FasterTransformer.

            - `beam_search_version(str, optional)`: Indicating the strategy of
            beam search. It can be 'v1' or 'v2'. 'v2' would select the top
            `beam_size * 2` beams and process the top `beam_size` alive and
            finish beams in them separately, while 'v1' would only select the
            top `beam_size` beams and mix up the alive and finish beams. 'v2' always
            searchs more and get better results, since the alive beams would
            always be `beam_size` while the number of alive beams in `v1` might
            decrease when meeting the end token. However, 'v2' always generates
            longer results thus might do more calculation and be slower.

            - `rel_len(bool, optional)`: Indicating whether `max_out_len` in is
            the length relative to that of source text. Only works in `v2` temporarily.
            It is suggest to set a small `max_out_len` and use `rel_len=True`.
            Default to False if not set.

            - `alpha(float, optional)`: The power number in length penalty
            calculation. Refer to `GNMT <https://arxiv.org/pdf/1609.08144.pdf>`_.
            Only works in `v2` temporarily. Default to 0.6 if not set.
        
            - diversity_rate(float, optional): Refer to `A Simple, Fast Diverse
            Decoding Algorithm for Neural Generation <https://arxiv.org/abs/1611.08562>`_
            for details. Bigger `diversity_rate` would lead to more diversity.
            if `diversity_rate == 0` is equivalent to naive BeamSearch. Default
            to 0 if not set. **NOTE**: Only works when using FasterTransformer
            temporarily.
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
        # Only works for FasterTransformer.
        # TODO: original version supports diversity rate.
        diversity_rate = kwargs.pop("diversity_rate", 0.0)
        use_fp16_decoding = kwargs.pop("use_fp16_decoding", False)
        use_ft = kwargs.pop("use_ft", True)
        beam_search_version = kwargs.pop("beam_search_version", "v1")
        rel_len = kwargs.pop("rel_len", False)
        alpha = kwargs.pop("alpha", 0.6)

        if use_ft:
            try:
                decoding_strategy = ("beam_search_v2"
                                     if beam_search_version == "v2" else
                                     "beam_search")
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
                    diversity_rate=diversity_rate,
                    decoding_strategy=decoding_strategy,
                    use_fp16_decoding=use_fp16_decoding,
                    rel_len=rel_len,
                    alpha=alpha)
            except Exception:
                logger.warning(
                    "Exception occurs when using FasterTransformer. " \
                    "The original forward will be involved. ")
                if diversity_rate != 0:
                    logger.warning(
                        "diversity_rate would not work since it is only " \
                        "supported by FasterTransformer temporarily.")
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
                    beam_search_version=beam_search_version,
                    rel_len=rel_len,
                    alpha=alpha)
        else:
            if diversity_rate != 0:
                logger.warning(
                    "diversity_rate would not work since it is only " \
                    "supported by FasterTransformer temporarily.")
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
                beam_search_version=beam_search_version,
                rel_len=rel_len,
                alpha=alpha)

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
                according to `output_time_major`. While, when using FasterTransformer
                and beam search v2, the beam dimension would be doubled to include
                both the top `beam_size` alive and finish beams, thus the tensor
                shape is `[batch_size, seq_len, beam_size * 2]` or `[seq_len, batch_size, beam_size * 2]`.
        
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


class FasterUnifiedTransformer(UnifiedTransformerPretrainedModel):
    def __init__(self,
                 model,
                 decoding_strategy="sampling",
                 decoding_lib=None,
                 use_fp16_decoding=False):
        super(FasterUnifiedTransformer, self).__init__()
        self._model = model
        self._decoding_strategy = decoding_strategy
        self.bos_token_id = model.bos_token_id
        self.pad_token_id = model.pad_token_id
        self.eos_token_id = model.eos_token_id
        self.unk_token_id = model.unk_token_id
        self.vocab_size = model.lm_head.decoder_bias.shape[0]
        self.logits_mask = self.generate_logits_mask(use_fp16_decoding)

        self._n_head = self._model.num_attention_heads
        self._hidden_dims = self._model.hidden_size
        self._normalize_before = self._model.normalize_before
        self._size_per_head = self._hidden_dims // self._n_head
        self._n_layer = self._model.num_hidden_layers
        self._mask_id = self._model.mask_token_id
        self._hidden_act = self._model.hidden_act

        self.decoding = InferUnifiedDecoding(
            model=self._model,
            decoding_strategy=self._decoding_strategy,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding,
            logits_mask=self.logits_mask,
            n_head=self._n_head,
            hidden_dims=self._hidden_dims,
            size_per_head=self._size_per_head,
            n_layer=self._n_layer,
            unk_id=self.unk_token_id,
            mask_id=self._mask_id,
            normalize_before=self._normalize_before,
            hidden_act=self._hidden_act)

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      token_type_ids,
                                      position_ids,
                                      attention_mask,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        input_ids = input_ids[:, :-1]
        decoding_type_id = token_type_ids[:, -1]
        token_type_ids = token_type_ids[:, :-1]
        position_ids = position_ids[:, :-1]
        attention_mask = attention_mask[:, :, :-1, :-1]
        seq_len = kwargs.get("seq_len", None) - 1

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
            "seq_len": seq_len,
            "decoding_type_id": paddle.cast(
                decoding_type_id, dtype="int32")
        }

    def generate_logits_mask(self, use_fp16_decoding):
        # pre-process distribution
        logits_mask = np.zeros(shape=[self.vocab_size], dtype=np.float32)
        logits_mask[self.unk_token_id] = -1e9
        logits_mask[self.bos_token_id] = -1e9
        logits_mask[self.pad_token_id] = -1e9

        logits_mask_t = paddle.assign(logits_mask)
        if use_fp16_decoding and self._decoding_strategy == "sampling":
            return paddle.cast(logits_mask_t, dtype="float16")
        else:
            return logits_mask_t

    def sample(self,
               input_ids,
               logits_processors,
               max_length,
               pad_token_id,
               eos_token_id,
               top_k=4,
               top_p=0.0,
               temperature=1.0,
               min_tokens_to_keep=1,
               **model_kwargs):
        max_length -= input_ids.shape[-1]
        model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                          **model_kwargs)

        if self._decoding_strategy == "sampling":
            if top_p == 1.0 and top_k > 0:
                top_p = 0.0
            elif top_p <= 0.0 and top_k == 0:
                raise ValueError(
                    "Topk sampling or topp sampling must be applied. " \
                    "Topk sampling and topp sampling cannot be both applied. ")
            elif (top_p > 0.0 and top_p < 1.0) and top_k > 0:
                raise ValueError(
                    "Topk sampling and topp sampling cannot be both applied. ")

        return self.forward(
            model_inputs=model_inputs,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature)

    def beam_search(self, input_ids, beam_scorer, logits_processors, max_length,
                    diversity_rate, pad_token_id, eos_token_id, **model_kwargs):
        max_length -= input_ids.shape[-1]
        model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                          **model_kwargs)
        temperature = model_kwargs.pop('temperature', 1.0)

        return self.forward(
            model_inputs=model_inputs,
            max_length=max_length,
            num_beams=beam_scorer.num_beams,
            diversity_rate=diversity_rate,
            temperature=temperature)

    def forward(self,
                max_length,
                decoding_strategy="sampling",
                top_k=4,
                top_p=0.0,
                num_beams=4,
                diversity_rate=0.0,
                temperature=1.0,
                model_inputs=None,
                **model_kwargs):
        seq_len = model_inputs.pop('seq_len', None)
        decoding_type_id = model_inputs.pop('decoding_type_id')

        outputs = self._model(**model_inputs)
        if isinstance(outputs, tuple):
            caches = outputs[1]
        else:
            raise RuntimeError('Not support.')
        cache_k = [c.k for c in caches]
        cache_v = [c.v for c in caches]

        return self.decoding(
            cache_k=cache_k,
            cache_v=cache_v,
            memory_seq_lens=seq_len,
            beam_size=num_beams,
            diversity_rate=diversity_rate,
            topk=top_k,
            topp=top_p,
            max_out_len=max_length,
            bos_id=self.bos_token_id,
            eos_id=self.eos_token_id,
            temperature=temperature,
            decoding_type_id=decoding_type_id,
            pos_bias=True)


class FasterUNIMOText(UNIMOPretrainedModel):
    def __init__(self,
                 model,
                 decoding_strategy="sampling",
                 decoding_lib=None,
                 use_fp16_decoding=False):
        super(FasterUNIMOText, self).__init__()
        self._model = model
        self._decoding_strategy = decoding_strategy
        self.bos_token_id = model.bos_token_id
        self.pad_token_id = model.pad_token_id
        self.eos_token_id = model.eos_token_id
        self.unk_token_id = model.unk_token_id
        self.vocab_size = model.lm_head.decoder_bias.shape[0]
        self.logits_mask = self.generate_logits_mask(use_fp16_decoding)

        self._n_head = self._model.num_attention_heads
        self._hidden_dims = self._model.hidden_size
        self._normalize_before = self._model.normalize_before
        self._size_per_head = self._hidden_dims // self._n_head
        self._n_layer = self._model.num_hidden_layers
        self._mask_id = self._model.mask_token_id
        self._hidden_act = self._model.hidden_act

        self.decoding = InferUnifiedDecoding(
            model=self._model,
            decoding_strategy=self._decoding_strategy,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding,
            logits_mask=self.logits_mask,
            n_head=self._n_head,
            hidden_dims=self._hidden_dims,
            size_per_head=self._size_per_head,
            n_layer=self._n_layer,
            unk_id=self.unk_token_id,
            mask_id=self._mask_id,
            normalize_before=self._normalize_before,
            hidden_act=self._hidden_act)

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      token_type_ids,
                                      position_ids,
                                      attention_mask,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        input_ids = input_ids[:, :-1]
        decoding_type_id = token_type_ids[:, -1]
        token_type_ids = token_type_ids[:, :-1]
        position_ids = position_ids[:, :-1]
        attention_mask = attention_mask[:, :, :-1, :-1]
        seq_len = kwargs.get("seq_len", None) - 1

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
            "seq_len": seq_len,
            "decoding_type_id": paddle.cast(
                decoding_type_id, dtype="int32")
        }

    def generate_logits_mask(self, use_fp16_decoding):
        # pre-process distribution
        logits_mask = np.zeros(shape=[self.vocab_size], dtype=np.float32)
        logits_mask[self.unk_token_id] = -1e9
        logits_mask[self.bos_token_id] = -1e9
        logits_mask[self.pad_token_id] = -1e9

        logits_mask_t = paddle.assign(logits_mask)
        if use_fp16_decoding and self._decoding_strategy == "sampling":
            return paddle.cast(logits_mask_t, dtype="float16")
        else:
            return logits_mask_t

    def sample(self,
               input_ids,
               logits_processors,
               max_length,
               pad_token_id,
               eos_token_id,
               top_k=4,
               top_p=0.0,
               temperature=1.0,
               min_tokens_to_keep=1,
               **model_kwargs):
        max_length -= input_ids.shape[-1]
        model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                          **model_kwargs)

        if self._decoding_strategy == "sampling":
            if top_p == 1.0 and top_k > 0:
                top_p = 0.0
            elif top_p <= 0.0 and top_k == 0:
                raise ValueError(
                    "Topk sampling or topp sampling must be applied. " \
                    "Topk sampling and topp sampling cannot be both applied. ")
            elif (top_p > 0.0 and top_p < 1.0) and top_k > 0:
                raise ValueError(
                    "Topk sampling and topp sampling cannot be both applied. ")

        return self.forward(
            model_inputs=model_inputs,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature)

    def beam_search(self, input_ids, beam_scorer, logits_processors, max_length,
                    diversity_rate, pad_token_id, eos_token_id, **model_kwargs):
        max_length -= input_ids.shape[-1]
        model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                          **model_kwargs)
        temperature = model_kwargs.pop('temperature', 1.0)

        return self.forward(
            model_inputs=model_inputs,
            max_length=max_length,
            num_beams=beam_scorer.num_beams,
            diversity_rate=diversity_rate,
            temperature=temperature)

    def forward(self,
                max_length,
                decoding_strategy="sampling",
                top_k=4,
                top_p=0.0,
                num_beams=4,
                diversity_rate=0.0,
                temperature=1.0,
                model_inputs=None,
                **model_kwargs):
        seq_len = model_inputs.pop('seq_len', None)
        decoding_type_id = model_inputs.pop('decoding_type_id')

        outputs = self._model(**model_inputs)
        if isinstance(outputs, tuple):
            caches = outputs[1]
        else:
            raise RuntimeError('Not support.')
        cache_k = [c.k for c in caches]
        cache_v = [c.v for c in caches]

        return self.decoding(
            cache_k=cache_k,
            cache_v=cache_v,
            memory_seq_lens=seq_len,
            beam_size=num_beams,
            diversity_rate=diversity_rate,
            topk=top_k,
            topp=top_p,
            max_out_len=max_length,
            bos_id=self.bos_token_id,
            eos_id=self.eos_token_id,
            temperature=temperature,
            decoding_type_id=decoding_type_id,
            pos_bias=False)


class FasterBART(BartPretrainedModel):
    def __init__(self,
                 model,
                 decoding_strategy="beam_search_v2",
                 decoding_lib=None,
                 use_fp16_decoding=False):
        super(FasterBART, self).__init__()
        self.use_fp16_decoding = use_fp16_decoding
        if use_fp16_decoding:
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Assign(
                model.bart.encoder.embed_tokens.weight))
            model.bart.encoder.embed_tokens = nn.Embedding(
                *model.bart.encoder.embed_tokens.weight.shape,
                weight_attr=weight_attr)
        self.encoder = model.bart.get_encoder()
        self.decoder = model.bart.get_decoder()
        self.bos_token_id = model.bart.config['bos_token_id']
        self.eos_token_id = model.bart.config['eos_token_id']
        self.pad_token_id = model.bart.config['pad_token_id']
        if decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v2"
        self._decoding_strategy = decoding_strategy
        self.decoding = InferBartDecoding(
            model=model,
            decoding_strategy=decoding_strategy,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def greedy_search(self, input_ids, logits_processors, max_length,
                      pad_token_id, eos_token_id, **model_kwargs):
        return self.sample(
            input_ids=input_ids,
            logits_processors=logits_processors,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            top_k=1,
            top_p=1.0,
            **model_kwargs)

    def beam_search(self, input_ids, beam_scorer, logits_processors, max_length,
                    diversity_rate, pad_token_id, eos_token_id, **model_kwargs):
        max_length -= input_ids.shape[-1]
        rel_len = model_kwargs.pop("rel_len", False)
        alpha = model_kwargs.pop("alpha", 0.6)
        encoder_output = model_kwargs.pop("encoder_output")
        mem_seq_lens = model_kwargs.pop("mem_seq_lens")
        return self.forward(
            encoder_output=encoder_output,
            mem_seq_lens=mem_seq_lens,
            beam_size=beam_scorer.num_beams,
            max_out_len=max_length,
            diversity_rate=diversity_rate,
            rel_len=rel_len,
            alpha=alpha)

    def sample(self,
               input_ids,
               logits_processors,
               max_length,
               pad_token_id,
               eos_token_id,
               top_k=4,
               top_p=0.0,
               temperature=1.0,
               min_tokens_to_keep=1,
               **model_kwargs):
        max_length -= input_ids.shape[-1]
        if self._decoding_strategy in ["sampling", "greedy_search"] and (
                abs(top_p - 1.0) < 1e-6 and top_k > 0):
            top_p = 0.0
        elif self._decoding_strategy == "sampling" and (top_p != 1.0 and
                                                        top_k == 0):
            top_k = 0
        else:
            raise ValueError(
                "Only top_k sampling or top_p sampling are supported. " \
                "Top_k sampling and top_p sampling cannot be both applied. ")
        encoder_output = model_kwargs.pop("encoder_output")
        mem_seq_lens = model_kwargs.pop("mem_seq_lens")
        return self.forward(
            encoder_output=encoder_output,
            mem_seq_lens=mem_seq_lens,
            top_k=top_k,
            top_p=top_p,
            max_out_len=max_length)

    def forward(self,
                input_ids=None,
                encoder_output=None,
                mem_seq_lens=None,
                beam_size=4,
                top_k=1,
                top_p=0.0,
                max_out_len=256,
                diversity_rate=0.0,
                rel_len=False,
                alpha=0.6):
        if encoder_output is None:
            assert input_ids is not None, "You have to specify either input_ids or encoder_output."
            encoder_output = self.encoder(input_ids)
        if mem_seq_lens is None:
            assert input_ids is not None, "You have to specify either input_ids when generating mem_seq_lens."
            mem_seq_lens = paddle.sum(paddle.cast(
                input_ids != self.pad_token_id, dtype="int32"),
                                      axis=-1,
                                      keepdim=True,
                                      dtype="int32")
        if self.use_fp16_decoding:
            encoder_output = paddle.cast(encoder_output, "float16")
        return self.decoding(
            enc_output=encoder_output,
            memory_seq_lens=mem_seq_lens,
            beam_size=beam_size,
            top_k=top_k,
            top_p=top_p,
            max_out_len=max_out_len,
            diversity_rate=diversity_rate,
            rel_len=rel_len,
            alpha=alpha)
