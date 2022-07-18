import os
import sys

import paddle
import paddle.nn.initializer as I

import paddle.nn as nn
import paddle.nn.functional as F
import config


def paddle2D_scatter_add(x_tensor, index_tensor, update_tensor, dim=0):
    dim0, dim1 = update_tensor.shape
    update_tensor = paddle.flatten(update_tensor, start_axis=0, stop_axis=1)
    index_tensor = paddle.reshape(index_tensor, [-1, 1])
    if dim == 0:
        index_tensor = paddle.concat(
            x=[index_tensor, (paddle.arange(dim1 * dim0) % dim0).unsqueeze(1)],
            axis=1)
    elif dim == 1:
        index_tensor = paddle.concat(x=[
            (paddle.arange(dim1 * dim0) // dim1).unsqueeze(1), index_tensor
        ],
                                     axis=1)
    output_tensor = paddle.scatter_nd_add(x_tensor, index_tensor, update_tensor)
    return output_tensor


class Encoder(paddle.nn.Layer):

    def __init__(self):
        super(Encoder, self).__init__()

        # Initialized embeddings
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.emb_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(
                std=config.trunc_norm_init_std)))

        # Initialized lstm weights
        self.lstm = nn.LSTM(
            config.emb_dim,
            config.hidden_dim,
            num_layers=1,
            direction='bidirect',
            weight_ih_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-config.rand_unif_init_mag,
                                      high=config.rand_unif_init_mag)),
            bias_ih_attr=paddle.ParamAttr(initializer=I.Constant(value=0.0)))

        # Initialized linear weights
        self.W_h = nn.Linear(config.hidden_dim * 2,
                             config.hidden_dim * 2,
                             bias_attr=False)

    # The variable seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        self.embedded = embedded

        output, hidden = self.lstm(embedded,
                                   sequence_length=paddle.to_tensor(
                                       seq_lens, dtype='int32'))

        encoder_feature = paddle.reshape(
            output, [-1, 2 * config.hidden_dim])  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return output, encoder_feature, hidden


class ReduceState(paddle.nn.Layer):

    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(
            config.hidden_dim * 2,
            config.hidden_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(
                std=config.trunc_norm_init_std)))
        self.reduce_c = nn.Linear(
            config.hidden_dim * 2,
            config.hidden_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(
                std=config.trunc_norm_init_std)))

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = paddle.reshape(h.transpose([1, 0, 2]),
                              [-1, config.hidden_dim * 2])
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = paddle.reshape(c.transpose([1, 0, 2]),
                              [-1, config.hidden_dim * 2])
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)
                )  # h, c dim = 1 x b x hidden_dim


class Attention(paddle.nn.Layer):

    def __init__(self):
        super(Attention, self).__init__()
        # Attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias_attr=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2,
                                     config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias_attr=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature,
                enc_padding_mask, coverage):
        b, t_k, n = encoder_outputs.shape

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = paddle.expand(dec_fea.unsqueeze(1),
                                         [b, t_k, n])  # B x t_k x 2*hidden_dim
        dec_fea_expanded = paddle.reshape(dec_fea_expanded,
                                          [-1, n])  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = paddle.reshape(coverage, [-1, 1])  # B * t_k x 1
            coverage_feature = self.W_c(
                coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = paddle.reshape(scores, [-1, t_k])  # B x t_k

        attn_dist_ = F.softmax(scores, axis=1) * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        # attn_dist = attn_dist_ / normalization_factor
        attn_dist = attn_dist_ / (
            paddle.reshape(normalization_factor, [-1, 1]) +
            paddle.ones_like(paddle.reshape(normalization_factor, [-1, 1])) *
            sys.float_info.epsilon)
        # See the issue: https://github.com/atulkum/pointer_summarizer/issues/54

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = paddle.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = paddle.reshape(c_t,
                             [-1, config.hidden_dim * 2])  # B x 2*hidden_dim

        attn_dist = paddle.reshape(attn_dist, [-1, t_k])  # B x t_k

        if config.is_coverage:
            coverage = paddle.reshape(coverage, [-1, t_k])
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(paddle.nn.Layer):

    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # Decoder
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.emb_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(
                std=config.trunc_norm_init_std)))

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim,
                                   config.emb_dim)

        self.lstm = nn.LSTM(
            config.emb_dim,
            config.hidden_dim,
            num_layers=1,
            direction='forward',
            weight_ih_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-config.rand_unif_init_mag,
                                      high=config.rand_unif_init_mag)),
            bias_ih_attr=paddle.ParamAttr(initializer=I.Constant(value=0.0)))

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(
                config.hidden_dim * 4 + config.emb_dim, 1)

        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim,
                              config.vocab_size,
                              weight_attr=paddle.ParamAttr(initializer=I.Normal(
                                  std=config.trunc_norm_init_std)))

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature,
                enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab,
                coverage, step):
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = paddle.concat(
                (paddle.reshape(h_decoder, [-1, config.hidden_dim]),
                 paddle.reshape(c_decoder, [-1, config.hidden_dim])),
                1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(
                s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask,
                coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(paddle.concat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = paddle.concat(
            (paddle.reshape(h_decoder, [-1, config.hidden_dim]),
             paddle.reshape(c_decoder, [-1, config.hidden_dim])),
            1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(
            s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask,
            coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = paddle.concat((c_t, s_t_hat, x),
                                        1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = paddle.concat(
            (paddle.reshape(lstm_out, [-1, config.hidden_dim]), c_t),
            1)  # B x hidden_dim * 3
        output1 = self.out1(output)  # B x hidden_dim
        output2 = self.out2(output1)  # B x vocab_size
        vocab_dist = F.softmax(output2, axis=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = paddle.concat([vocab_dist_, extra_zeros], 1)
            final_dist = paddle2D_scatter_add(vocab_dist_,
                                              enc_batch_extend_vocab,
                                              attn_dist_, 1)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Model(object):

    def __init__(self, model_file_path=None, is_eval=False):
        super(Model, self).__init__()
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # Shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight

        if is_eval:
            encoder.eval()
            decoder.eval()
            reduce_state.eval()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            self.decoder.set_state_dict(
                paddle.load(os.path.join(model_file_path, 'decoder.params')))
            self.encoder.set_state_dict(
                paddle.load(os.path.join(model_file_path, 'encoder.params')))
            self.reduce_state.set_state_dict(
                paddle.load(os.path.join(model_file_path,
                                         'reduce_state.params')))
