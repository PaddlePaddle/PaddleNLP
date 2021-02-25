import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import TransformerModel
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
                 beam_size=4,
                 max_out_len=256):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.beam_size = args.pop("beam_size")
        self.max_out_len = args.pop("max_out_len")
        self.dropout = dropout
        self.weight_sharing = weight_sharing
        self.trg_vocab_size = trg_vocab_size
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
            beam_size=beam_size,
            max_out_len=max_out_len)

    def forward(self, src_word):
        if self.weight_sharing:
            self.decoding_linear.weight.set_value(
                self.trg_word_embedding.word_embedding.weight.t())
            self.decoding_linear.bias = paddle.create_parameter(
                shape=[self.trg_vocab_size],
                dtype=paddle.get_default_dtype(),
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(value=0.0))
        else:
            self.decoding_linear = self.linear

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

        mem_seq_lens = paddle.sum(paddle.cast(
            src_word != self.bos_id, dtype="int32"),
                                  axis=1)
        ids = self.decoding(enc_output, mem_seq_lens)

        return ids
