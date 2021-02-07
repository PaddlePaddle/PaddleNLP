from __future__ import print_function

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.layers.utils import map_structure


def position_encoding_init(n_position, d_pos_vec, dtype="float32"):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(
        np.arange(num_timescales) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype(dtype)


class WordEmbedding(nn.Layer):
    """
    Word Embedding + Scale
    """

    def __init__(self, vocab_size, emb_dim, bos_idx=0):
        super(WordEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=bos_idx,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0., emb_dim**-0.5)))

    def forward(self, word):
        word_emb = self.emb_dim**0.5 * self.word_embedding(word)
        return word_emb


class PositionalEmbedding(nn.Layer):
    """
    Positional Embedding
    """

    def __init__(self, emb_dim, max_length, bos_idx=0):
        super(PositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.pos_encoder = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=self.emb_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    position_encoding_init(max_length, self.emb_dim))))

    def forward(self, pos):
        pos_emb = self.pos_encoder(pos)
        pos_emb.stop_gradient = True
        return pos_emb


class CrossEntropyCriterion(nn.Layer):
    def __init__(self, label_smooth_eps, pad_idx=0):
        super(CrossEntropyCriterion, self).__init__()
        self.label_smooth_eps = label_smooth_eps
        self.pad_idx = pad_idx

    def forward(self, predict, label):
        weights = paddle.cast(
            label != self.pad_idx, dtype=paddle.get_default_dtype())
        if self.label_smooth_eps:
            label = paddle.squeeze(label, axis=[2])
            label = F.label_smooth(
                label=F.one_hot(
                    x=label, num_classes=predict.shape[-1]),
                epsilon=self.label_smooth_eps)

        cost = F.softmax_with_cross_entropy(
            logits=predict,
            label=label,
            soft_label=True if self.label_smooth_eps else False)
        weighted_cost = cost * weights
        sum_cost = paddle.sum(weighted_cost)
        token_num = paddle.sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, token_num


class TransformerDecodeCell(nn.Layer):
    def __init__(self,
                 decoder,
                 word_embedding=None,
                 pos_embedding=None,
                 linear=None,
                 dropout=0.1):
        super(TransformerDecodeCell, self).__init__()
        self.decoder = decoder
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding
        self.linear = linear
        self.dropout = dropout

    def forward(self, inputs, states, static_cache, trg_src_attn_bias, memory):
        if states and static_cache:
            states = list(zip(states, static_cache))

        if self.word_embedding:
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs)

            word_emb = self.word_embedding(inputs[0])
            pos_emb = self.pos_embedding(inputs[1])
            word_emb = word_emb + pos_emb
            inputs = F.dropout(
                word_emb, p=self.dropout,
                training=False) if self.dropout else word_emb

            cell_outputs, new_states = self.decoder(inputs, memory, None,
                                                    trg_src_attn_bias, states)
        else:
            cell_outputs, new_states = self.decoder(inputs, memory, None,
                                                    trg_src_attn_bias, states)

        if self.linear:
            cell_outputs = self.linear(cell_outputs)

        new_states = [cache[0] for cache in new_states]

        return cell_outputs, new_states


class TransformerBeamSearchDecoder(nn.decode.BeamSearchDecoder):
    def __init__(self, cell, start_token, end_token, beam_size,
                 var_dim_in_state):
        super(TransformerBeamSearchDecoder,
              self).__init__(cell, start_token, end_token, beam_size)
        self.cell = cell
        self.var_dim_in_state = var_dim_in_state

    def _merge_batch_beams_with_var_dim(self, c):
        # Init length of cache is 0, and it increases with decoding carrying on,
        # thus need to reshape elaborately
        var_dim_in_state = self.var_dim_in_state + 1  # count in beam dim
        c = paddle.transpose(c,
                             list(range(var_dim_in_state, len(c.shape))) +
                             list(range(0, var_dim_in_state)))
        c = paddle.reshape(
            c, [0] * (len(c.shape) - var_dim_in_state
                      ) + [self.batch_size * self.beam_size] +
            [int(size) for size in c.shape[-var_dim_in_state + 2:]])
        c = paddle.transpose(
            c,
            list(range((len(c.shape) + 1 - var_dim_in_state), len(c.shape))) +
            list(range(0, (len(c.shape) + 1 - var_dim_in_state))))
        return c

    def _split_batch_beams_with_var_dim(self, c):
        var_dim_size = paddle.shape(c)[self.var_dim_in_state]
        c = paddle.reshape(
            c, [-1, self.beam_size] +
            [int(size)
             for size in c.shape[1:self.var_dim_in_state]] + [var_dim_size] +
            [int(size) for size in c.shape[self.var_dim_in_state + 1:]])
        return c

    @staticmethod
    def tile_beam_merge_with_batch(t, beam_size):
        return map_structure(
            lambda x: nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(x, beam_size),
            t)

    def step(self, time, inputs, states, **kwargs):
        # Steps for decoding.
        # Compared to RNN, Transformer has 3D data at every decoding step
        inputs = paddle.reshape(inputs, [-1, 1])  # token
        pos = paddle.ones_like(inputs) * time  # pos

        cell_states = map_structure(self._merge_batch_beams_with_var_dim,
                                    states.cell_states)

        cell_outputs, next_cell_states = self.cell((inputs, pos), cell_states,
                                                   **kwargs)

        # Squeeze to adapt to BeamSearchDecoder which use 2D logits
        cell_outputs = map_structure(
            lambda x: paddle.squeeze(x, [1]) if len(x.shape) == 3 else x,
            cell_outputs)
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams_with_var_dim,
                                         next_cell_states)

        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,
            next_cell_states=next_cell_states,
            beam_state=states)
        next_inputs, finished = (beam_search_output.predicted_ids,
                                 beam_search_state.finished)

        return (beam_search_output, beam_search_state, next_inputs, finished)


class TransformerModel(nn.Layer):
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
                 eos_id=1):
        super(TransformerModel, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout

        self.src_word_embedding = WordEmbedding(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_idx=self.bos_id)
        self.src_pos_embedding = PositionalEmbedding(
            emb_dim=d_model, max_length=max_length, bos_idx=self.bos_id)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            self.trg_word_embedding = self.src_word_embedding
            self.trg_pos_embedding = self.src_pos_embedding
        else:
            self.trg_word_embedding = WordEmbedding(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_idx=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(
                emb_dim=d_model, max_length=max_length, bos_idx=self.bos_id)

        self.transformer = paddle.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            activation="relu",
            normalize_before=True)

        if weight_sharing:
            self.linear = lambda x: paddle.matmul(x=x,
                                                  y=self.trg_word_embedding.word_embedding.weight,
                                                  transpose_y=True)
        else:
            self.linear = nn.Linear(
                in_features=d_model,
                out_features=trg_vocab_size,
                bias_attr=False)

    def forward(self, src_word, trg_word):
        src_max_len = paddle.shape(src_word)[-1]
        trg_max_len = paddle.shape(trg_word)[-1]
        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        src_slf_attn_bias.stop_gradient = True
        trg_slf_attn_bias = self.transformer.generate_square_subsequent_mask(
            trg_max_len)
        trg_slf_attn_bias.stop_gradient = True
        trg_src_attn_bias = src_slf_attn_bias
        src_pos = paddle.cast(
            src_word != self.bos_id, dtype="int64") * paddle.arange(
                start=0, end=src_max_len)
        trg_pos = paddle.cast(
            trg_word != self.bos_id, dtype="int64") * paddle.arange(
                start=0, end=trg_max_len)
        with paddle.static.amp.fp16_guard():
            src_emb = self.src_word_embedding(src_word)
            src_pos_emb = self.src_pos_embedding(src_pos)
            src_emb = src_emb + src_pos_emb
            enc_input = F.dropout(
                src_emb, p=self.dropout,
                training=self.training) if self.dropout else src_emb

            trg_emb = self.trg_word_embedding(trg_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb

            dec_output = self.transformer(
                enc_input,
                dec_input,
                src_mask=src_slf_attn_bias,
                tgt_mask=trg_slf_attn_bias,
                memory_mask=trg_src_attn_bias)

            predict = self.linear(dec_output)

        return predict


class InferTransformerModel(TransformerModel):
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
        super(InferTransformerModel, self).__init__(**args)

        cell = TransformerDecodeCell(
            self.transformer.decoder, self.trg_word_embedding,
            self.trg_pos_embedding, self.linear, self.dropout)

        self.decode = TransformerBeamSearchDecoder(
            cell, bos_id, eos_id, beam_size, var_dim_in_state=2)

    def forward(self, src_word):
        src_max_len = paddle.shape(src_word)[-1]
        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        trg_src_attn_bias = src_slf_attn_bias
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

        # Init states (caches) for transformer, need to be updated according to selected beam
        incremental_cache, static_cache = self.transformer.decoder.gen_cache(
            enc_output, do_zip=True)

        static_cache, enc_output, trg_src_attn_bias = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
            (static_cache, enc_output, trg_src_attn_bias), self.beam_size)

        rs, _ = nn.decode.dynamic_decode(
            decoder=self.decode,
            inits=incremental_cache,
            max_step_num=self.max_out_len,
            memory=enc_output,
            trg_src_attn_bias=trg_src_attn_bias,
            static_cache=static_cache,
            is_test=True)

        return rs
