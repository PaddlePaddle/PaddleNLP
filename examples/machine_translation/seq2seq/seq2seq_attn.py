# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as I


class CrossEntropyCriterion(nn.Layer):

    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, predict, label, trg_mask):
        cost = F.cross_entropy(input=predict,
                               label=label,
                               soft_label=False,
                               reduction='none')
        cost = paddle.squeeze(cost, axis=[2])
        masked_cost = cost * trg_mask
        batch_mean_cost = paddle.mean(masked_cost, axis=[0])
        seq_cost = paddle.sum(batch_mean_cost)

        return seq_cost


class Seq2SeqEncoder(nn.Layer):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(Seq2SeqEncoder, self).__init__()
        self.embedder = nn.Embedding(
            vocab_size,
            embed_dim,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            direction="forward",
                            dropout=dropout_prob if num_layers > 1 else 0.)

    def forward(self, sequence, sequence_length):
        inputs = self.embedder(sequence)
        encoder_output, encoder_state = self.lstm(
            inputs, sequence_length=sequence_length)

        return encoder_output, encoder_state


class AttentionLayer(nn.Layer):

    def __init__(self, hidden_size, bias=False, init_scale=0.1):
        super(AttentionLayer, self).__init__()
        self.input_proj = nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)),
            bias_attr=bias)
        self.output_proj = nn.Linear(
            hidden_size + hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)),
            bias_attr=bias)

    def forward(self, hidden, encoder_output, encoder_padding_mask):
        encoder_output = self.input_proj(encoder_output)
        attn_scores = paddle.matmul(paddle.unsqueeze(hidden, [1]),
                                    encoder_output,
                                    transpose_y=True)

        if encoder_padding_mask is not None:
            attn_scores = paddle.add(attn_scores, encoder_padding_mask)

        attn_scores = F.softmax(attn_scores)
        attn_out = paddle.squeeze(paddle.matmul(attn_scores, encoder_output),
                                  [1])
        attn_out = paddle.concat([attn_out, hidden], 1)
        attn_out = self.output_proj(attn_out)
        return attn_out


class Seq2SeqDecoderCell(nn.RNNCellBase):

    def __init__(self, num_layers, input_size, hidden_size, dropout_prob=0.):
        super(Seq2SeqDecoderCell, self).__init__()
        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

        self.lstm_cells = nn.LayerList([
            nn.LSTMCell(input_size=input_size +
                        hidden_size if i == 0 else hidden_size,
                        hidden_size=hidden_size) for i in range(num_layers)
        ])

        self.attention_layer = AttentionLayer(hidden_size)

    def forward(self,
                step_input,
                states,
                encoder_output,
                encoder_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = paddle.concat([step_input, input_feed], 1)
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_lstm_state = lstm_cell(step_input, lstm_states[i])
            if self.dropout:
                step_input = self.dropout(out)
            else:
                step_input = out

            new_lstm_states.append(new_lstm_state)
        out = self.attention_layer(step_input, encoder_output,
                                   encoder_padding_mask)
        return out, [new_lstm_states, out]


class Seq2SeqDecoder(nn.Layer):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(Seq2SeqDecoder, self).__init__()
        self.embedder = nn.Embedding(
            vocab_size,
            embed_dim,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))
        self.lstm_attention = nn.RNN(Seq2SeqDecoderCell(num_layers, embed_dim,
                                                        hidden_size,
                                                        dropout_prob),
                                     is_reverse=False,
                                     time_major=False)
        self.output_layer = nn.Linear(
            hidden_size,
            vocab_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)),
            bias_attr=False)

    def forward(self, trg, decoder_initial_states, encoder_output,
                encoder_padding_mask):
        inputs = self.embedder(trg)

        decoder_output, _ = self.lstm_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        predict = self.output_layer(decoder_output)

        return predict


class Seq2SeqAttnModel(nn.Layer):

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 eos_id=1,
                 init_scale=0.1):
        super(Seq2SeqAttnModel, self).__init__()
        self.hidden_size = hidden_size
        self.eos_id = eos_id
        self.num_layers = num_layers
        self.INF = 1e9
        self.encoder = Seq2SeqEncoder(src_vocab_size, embed_dim, hidden_size,
                                      num_layers, dropout_prob, init_scale)
        self.decoder = Seq2SeqDecoder(trg_vocab_size, embed_dim, hidden_size,
                                      num_layers, dropout_prob, init_scale)

    def forward(self, src, src_length, trg):
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        # Transfer shape of encoder_final_states to [num_layers, 2, batch_size, hidden_size]
        encoder_final_states = [(encoder_final_state[0][i],
                                 encoder_final_state[1][i])
                                for i in range(self.num_layers)]
        # Construct decoder initial states: use input_feed and the shape is
        # [[h,c] * num_layers, input_feed], consistent with Seq2SeqDecoderCell.states
        decoder_initial_states = [
            encoder_final_states,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # Build attention mask to avoid paying attention on padddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())
        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])

        predict = self.decoder(trg, decoder_initial_states, encoder_output,
                               encoder_padding_mask)
        return predict


class Seq2SeqAttnInferModel(Seq2SeqAttnModel):

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.bos_id = args.pop("bos_id")
        self.beam_size = args.pop("beam_size")
        self.max_out_len = args.pop("max_out_len")
        self.num_layers = num_layers
        super(Seq2SeqAttnInferModel, self).__init__(**args)
        # Dynamic decoder for inference
        self.beam_search_decoder = nn.BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)

    def forward(self, src, src_length):
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        encoder_final_state = [(encoder_final_state[0][i],
                                encoder_final_state[1][i])
                               for i in range(self.num_layers)]

        # Initial decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # Build attention mask to avoid paying attention on paddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])

        # Tile the batch dimension with beam_size
        encoder_output = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, self.beam_size)
        encoder_padding_mask = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_padding_mask, self.beam_size)

        # Dynamic decoding with beam search
        seq_output, _ = nn.dynamic_decode(
            decoder=self.beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=self.max_out_len,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        return seq_output
