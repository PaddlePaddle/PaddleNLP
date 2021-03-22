import os
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import TransformerModel, position_encoding_init
from paddlenlp.ext_op import InferTransformerDecoding


class FasterTransformer(TransformerModel):
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
                 decoding_strategy="beam_search",
                 beam_size=4,
                 topk=1,
                 topp=0.0,
                 max_out_len=256,
                 decoding_lib=None,
                 use_fp16_decoding=False):
        if decoding_lib is None:
            raise ValueError(
                "The args decoding_lib must be set to use Faster Transformer. ")
        elif not os.path.exists(decoding_lib):
            raise ValueError("The path to decoding lib is not exist.")

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

        self.decoding = InferTransformerDecoding(
            decoder=self.transformer.decoder,
            word_embedding=self.trg_word_embedding.word_embedding,
            positional_embedding=self.trg_pos_embedding.pos_encoder,
            linear=self.decoding_linear,
            max_length=max_length,
            n_layer=n_layer,
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
            src_word != self.bos_id, dtype="int64") * paddle.arange(
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
                                  axis=1)
        ids = self.decoding(enc_output, mem_seq_lens)

        return ids

    def load(self, init_from_params):
        # Load the trained model
        assert init_from_params, (
            "Please set init_from_params to load the infer model.")

        model_dict = paddle.load(init_from_params)

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

        self.load_dict(model_dict)
