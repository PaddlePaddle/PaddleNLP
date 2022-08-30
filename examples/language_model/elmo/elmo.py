import numpy as np
from typing import List
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as I

from dataset import load_vocab, create_batches


def reverse_sequence(x, sequence_lengths):
    batch_size = x.shape[0]
    sequence_lengths = sequence_lengths.numpy().data
    y = paddle.zeros(x.shape, x.dtype)
    for i in range(batch_size):
        lens = sequence_lengths[i]
        z = x[i, :lens, :]
        z = paddle.reverse(z, axis=[0])
        y[i, :lens, :] = z
    return y


class ELMo(nn.Layer):

    def __init__(self,
                 batch_size=None,
                 char_embed_dim=16,
                 projection_dim=512,
                 vocab_size=None,
                 cnn_filters=[[1, 32], [2, 32], [3, 64], [4, 128], [5, 256],
                              [6, 512], [7, 1024]],
                 char_vocab_size=262,
                 max_characters_per_token=50,
                 num_highways=2,
                 num_layers=2,
                 dropout=0.1,
                 task='pre-train'):
        super(ELMo, self).__init__()

        if task == 'pre-train':
            if vocab_size is None or batch_size is None:
                raise ValueError(
                    'vocab_size and batch_size should be set when task="pre-train"'
                )
        elif task == 'fine-tune':
            if batch_size is None:
                batch_size = 128
        else:
            raise ValueError('task should be "pre-train" or "fine-tune"')

        self._projection_dim = projection_dim
        self._task = task

        self._token_embding_layer = ELMoCharacterEncoderLayer(
            char_vocab_size, char_embed_dim, projection_dim, num_highways,
            cnn_filters, max_characters_per_token)
        self._elmobilm = ELMoBiLM(batch_size, projection_dim, projection_dim,
                                  num_layers, dropout, task)
        if task == 'pre-train':
            paramAttr = paddle.ParamAttr(initializer=I.Normal(
                mean=0.0, std=1.0 / np.sqrt(projection_dim)))
            self._linear_layer = nn.Linear(projection_dim,
                                           vocab_size,
                                           weight_attr=paramAttr)

    @property
    def embedding_dim(self):
        return self._projection_dim * 2

    def forward(self, inputs):
        # [batch_size, seq_len, max_characters_per_token]
        ids, ids_reverse = inputs
        # [batch_size, seq_len, projection_dim]
        token_embedding = self._token_embding_layer(ids)
        token_embedding_reverse = self._token_embding_layer(ids_reverse)

        outs = self._elmobilm(token_embedding, token_embedding_reverse)

        if self._task == 'pre-train':
            # [batch_size, seq_len, projection_dim]
            fw_out, bw_out = outs

            # [batch_size, max_seq_len, vocab_size]
            fw_logits = self._linear_layer(fw_out)
            bw_logits = self._linear_layer(bw_out)
            return [fw_logits, bw_logits]
        else:
            mask = paddle.any(ids > 0, axis=2)
            seq_lens = paddle.sum(paddle.cast(mask, dtype=ids.dtype), axis=1)
            outputs = [
                paddle.concat([token_embedding, token_embedding], axis=2)
            ]
            for fw_h, bw_h in zip(outs[0], outs[1]):
                bw_h = reverse_sequence(bw_h, seq_lens)
                outputs.append(paddle.concat([fw_h, bw_h], axis=2))
            # [batch_size, num_lstm_layers + 1, max_seq_len, projection_dim * 2]
            outputs = paddle.concat(
                [paddle.unsqueeze(emb, axis=1) for emb in outputs], axis=1)
            return outputs


class ELMoBiLM(nn.Layer):

    def __init__(self,
                 batch_size,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 task='pre-train'):
        super(ELMoBiLM, self).__init__()

        self._num_layers = num_layers
        self._dropout = dropout
        self._task = task

        self._lstm_layers = []
        for direction in ['forward', 'backward']:
            layers = []
            for i in range(num_layers):
                lstm = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               direction='forward',
                               weight_hh_attr=paddle.ParamAttr(
                                   initializer=I.XavierUniform()),
                               weight_ih_attr=paddle.ParamAttr(
                                   initializer=I.XavierUniform()),
                               bias_hh_attr=False,
                               bias_ih_attr=paddle.ParamAttr(
                                   initializer=I.Constant(value=0.0)))
                self.add_sublayer('{}_lstm_layer_{}'.format(direction, i), lstm)

                hidden_state = paddle.zeros(shape=[1, batch_size, hidden_size],
                                            dtype='float32')
                cell_state = paddle.zeros(shape=[1, batch_size, hidden_size],
                                          dtype='float32')
                layers.append({
                    'lstm': lstm,
                    'hidden_state': hidden_state,
                    'cell_state': cell_state
                })
            self._lstm_layers.append(layers)

        if dropout:
            self._dropout_layer = nn.Dropout(p=dropout)

    def forward(self, fw_x, bw_x):
        final_outs = []
        lstm_outs = []
        for x, layers in zip([fw_x, bw_x], self._lstm_layers):
            batch_size = x.shape[0]
            outs = []
            for i, dic in enumerate(layers):
                lstm = dic['lstm']
                hidden_state = dic['hidden_state'][:, :batch_size, :]
                cell_state = dic['cell_state'][:, :batch_size, :]
                if self._dropout:
                    x = self._dropout_layer(x)
                x, (hidden_state, cell_state) = lstm(x,
                                                     (hidden_state, cell_state))
                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()
                dic['hidden_state'][:, :batch_size, :] = hidden_state
                dic['cell_state'][:, :batch_size, :] = cell_state
                outs.append(x)
            lstm_outs.append(outs)

            if self._dropout:
                x = self._dropout_layer(x)
            final_outs.append(x)
        if self._task == 'pre-train':
            return final_outs
        else:
            return lstm_outs


class ELMoCharacterEncoderLayer(nn.Layer):

    def __init__(self, char_vocab_size, char_embed_dim, projection_dim,
                 num_highways, cnn_filters, max_characters_per_token):
        super(ELMoCharacterEncoderLayer, self).__init__()

        self._use_highway = (num_highways > 0)
        self._n_filters = sum(f[1] for f in cnn_filters)
        self._use_proj = (self._n_filters != projection_dim)

        paramAttr = paddle.ParamAttr(initializer=I.Uniform(low=-1.0, high=1.0))
        self._char_embedding_layer = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_embed_dim,
            weight_attr=paramAttr)
        self._char_embedding_layer.weight[0, :] = 0

        self._convolution_layers = []
        for i, (width, num) in enumerate(cnn_filters):
            paramAttr = paddle.ParamAttr(
                initializer=I.Uniform(low=-0.05, high=0.05))
            conv2d = nn.Conv2D(in_channels=char_embed_dim,
                               out_channels=num,
                               kernel_size=(1, width),
                               padding='Valid',
                               data_format='NHWC',
                               weight_attr=paramAttr)
            max_pool = nn.MaxPool2D(kernel_size=(1, max_characters_per_token -
                                                 width + 1),
                                    stride=(1, 1),
                                    padding='Valid',
                                    data_format='NHWC')
            self.add_sublayer('cnn_layer_{}'.format(i), conv2d)
            self.add_sublayer('maxpool_layer_{}'.format(i), max_pool)
            self._convolution_layers.append([width, conv2d, max_pool])

        self._relu = nn.ReLU()
        if self._use_highway:
            self._highway_layer = Highway(self._n_filters, num_highways)
        if self._use_proj:
            paramAttr = paddle.ParamAttr(initializer=I.Normal(
                mean=0.0, std=1.0 / np.sqrt(self._n_filters)))
            self._linear_layer = nn.Linear(self._n_filters,
                                           projection_dim,
                                           weight_attr=paramAttr)

    def forward(self, x):
        # [batch_size, seq_len, max_characters_per_token, embed_dim]
        char_embedding = self._char_embedding_layer(x)

        cnn_outs = []
        for width, conv2d, max_pool in self._convolution_layers:
            # [batch_size, seq_len, max_characters_per_token - kerner_width, out_channel]
            conv_out = conv2d(char_embedding)
            # [batch_size, seq_len, 1, out_channel]
            pool_out = max_pool(conv_out)
            # [batch_size, seq_len, 1, out_channel]
            out = self._relu(pool_out)
            # [batch_size, seq_len, out_channel]
            out = paddle.squeeze(out, axis=2)
            cnn_outs.append(out)

        # [batch_size, seq_len, n_filters]
        token_embedding = paddle.concat(cnn_outs, axis=-1)

        if self._use_highway:
            # [batch_size, seq_len, n_filters]
            token_embedding = self._highway_layer(token_embedding)

        if self._use_proj:
            # [batch_size, seq_len, projection_dim]
            token_embedding = self._linear_layer(token_embedding)

        return token_embedding


class Highway(nn.Layer):

    def __init__(self, input_dim, num_layers):
        super(Highway, self).__init__()

        self._num_layers = num_layers

        self._highway_layers = []
        for i in range(num_layers):
            paramAttr = paddle.ParamAttr(
                initializer=I.Normal(mean=0.0, std=1.0 / np.sqrt(input_dim)))
            paramAttr_b = paddle.ParamAttr(initializer=I.Constant(value=-2.0))
            carry_linear = nn.Linear(input_dim,
                                     input_dim,
                                     weight_attr=paramAttr,
                                     bias_attr=paramAttr_b)
            self.add_sublayer('carry_linear_{}'.format(i), carry_linear)

            paramAttr = paddle.ParamAttr(
                initializer=I.Normal(mean=0.0, std=1.0 / np.sqrt(input_dim)))
            transform_linear = nn.Linear(input_dim,
                                         input_dim,
                                         weight_attr=paramAttr)
            self.add_sublayer('transform_linear_{}'.format(i), transform_linear)

            self._highway_layers.append([carry_linear, transform_linear])

        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self._num_layers):
            carry_linear, transform_linear = self._highway_layers[i]
            carry_gate = self._sigmoid(carry_linear(x))
            transform_gate = self._relu(transform_linear(x))
            x = carry_gate * transform_gate + (1.0 - carry_gate) * x
        return x


class ELMoLoss(nn.Layer):

    def __init__(self):
        super(ELMoLoss, self).__init__()

    def forward(self, x, y):
        # [batch_size, seq_len, vocab_size]
        fw_logits, bw_logits = x
        # [batch_size, seq_len]
        fw_label, bw_label = y
        # [batch_size, seq_len, 1]
        fw_label = paddle.unsqueeze(fw_label, axis=2)
        bw_label = paddle.unsqueeze(bw_label, axis=2)

        # [batch_size, seq_len, 1]
        fw_loss = F.cross_entropy(input=fw_logits, label=fw_label)
        bw_loss = F.cross_entropy(input=bw_logits, label=bw_label)

        avg_loss = 0.5 * (fw_loss + bw_loss)
        return avg_loss


def get_elmo_layer(params_file, batch_size, trainable=False):
    if trainable:
        elmo = ELMo(batch_size=batch_size, task='fine-tune')
    else:
        elmo = ELMo(batch_size=batch_size, dropout=None, task='fine-tune')
    weight_state_dict = paddle.load(params_file + '.pdparams')
    elmo.set_state_dict(weight_state_dict)
    if trainable:
        elmo.train()
    else:
        for params in elmo.parameters():
            params.trainable = False
        elmo.eval()
    return elmo


class ELMoEmbedder(object):

    def __init__(self, params_file, batch_size=128, max_seq_len=256):
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size

        self._elmo = get_elmo_layer(params_file, batch_size, trainable=False)
        self._vocab = load_vocab()

    def encode(self, sentences: List[List[str]]):
        """
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        """
        batch_data = create_batches(sentences, self._batch_size, self._vocab,
                                    self._max_seq_len)
        embeddings = []
        for data in batch_data:
            ids, ids_reverse, seq_lens = data
            # [batch_size, num_lstm_layers + 1, max_seq_len, projection_dim * 2]
            outputs = self._elmo([ids, ids_reverse])
            outputs = outputs.numpy()
            for i, lens in enumerate(seq_lens):
                arr = outputs[i, :, 1:lens - 1, :]
                embeddings.append(arr)
        return embeddings
