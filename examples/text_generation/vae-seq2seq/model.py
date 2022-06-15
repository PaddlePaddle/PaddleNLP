#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as I


class CrossEntropyWithKL(nn.Layer):
    """
    backward_loss = kl_loss * kl_weight + cross_entropy_loss
    """

    def __init__(self, base_kl_weight, anneal_r):
        super(CrossEntropyWithKL, self).__init__()
        self.kl_weight = base_kl_weight
        self.anneal_r = anneal_r
        self.loss = 0.0
        self.kl_loss = 0.0
        self.rec_loss = 0.0

    def update_kl_weight(self):
        self.kl_weight = min(1.0, self.kl_weight + self.anneal_r)

    def forward(self, kl_loss, dec_output, trg_mask, label):
        self.update_kl_weight()
        self.kl_loss = kl_loss

        rec_loss = F.cross_entropy(input=dec_output,
                                   label=label,
                                   reduction='none',
                                   soft_label=False)

        rec_loss = paddle.squeeze(rec_loss, axis=[2])
        rec_loss = rec_loss * trg_mask
        rec_loss = paddle.mean(rec_loss, axis=[0])
        rec_loss = paddle.sum(rec_loss)
        self.rec_loss = rec_loss

        self.loss = self.kl_loss * self.kl_weight + self.rec_loss
        return self.loss


class Perplexity(paddle.metric.Metric):

    def __init__(self, name='ppl', reset_freq=100, *args, **kwargs):
        self.cross_entropy = kwargs.pop('loss')
        super(Perplexity, self).__init__(*args, **kwargs)
        self._name = name
        self.total_ce = 0
        self.word_count = 0
        self.reset_freq = reset_freq
        self.batch_size = 0

    def update(self, kl_loss, dec_output, trg_mask, label, *args):
        # Perplexity is calculated using cross entropy
        self.batch_size = dec_output.shape[0]
        loss = self.cross_entropy.loss.numpy()
        self.total_ce += loss[0] * self.batch_size
        self.word_count += np.sum(trg_mask)

    def reset(self):
        self.total_ce = 0
        self.word_count = 0

    def accumulate(self):
        return np.exp(self.total_ce / self.word_count)

    def name(self):
        return self._name


class NegativeLogLoss(paddle.metric.Metric):

    def __init__(self, name='nll', reset_freq=100, *args, **kwargs):
        self.cross_entropy = kwargs.pop('loss')
        super(NegativeLogLoss, self).__init__(*args, **kwargs)
        self._name = name
        self.total_ce = 0
        self.batch_count = 0
        self.reset_freq = reset_freq
        self.batch_size = 0
        self.sample_count = 0

    def update(self, kl_loss, dec_output, trg_mask, label, *args):
        self.batch_size = dec_output.shape[0]
        loss = self.cross_entropy.loss.numpy()
        self.total_ce += loss[0] * self.batch_size
        self.sample_count += self.batch_size

    def reset(self):
        self.total_ce = 0
        self.sample_count = 0

    def accumulate(self):
        return (self.total_ce / self.sample_count)

    def name(self):
        return self._name


class TrainCallback(paddle.callbacks.ProgBarLogger):

    def __init__(self, ppl, nll, log_freq=200, verbose=2):
        super(TrainCallback, self).__init__(log_freq, verbose)
        self.ppl = ppl
        self.nll = nll

    def on_train_begin(self, logs=None):
        super(TrainCallback, self).on_train_begin(logs)
        self.train_metrics = [
            "loss", "ppl", "nll", "kl weight", "kl loss", "rec loss"
        ]

    def on_epoch_begin(self, epoch=None, logs=None):
        super(TrainCallback, self).on_epoch_begin(epoch, logs)
        self.ppl.reset()
        self.nll.reset()

    def on_train_batch_end(self, step, logs=None):
        # loss and kl weight are not accumulated
        logs["kl weight"] = self.ppl.cross_entropy.kl_weight
        logs["kl loss"] = self.ppl.cross_entropy.kl_loss.numpy()[0]
        logs["rec loss"] = self.ppl.cross_entropy.rec_loss.numpy()[0]
        super(TrainCallback, self).on_train_batch_end(step, logs)

    def on_eval_begin(self, logs=None):
        super(TrainCallback, self).on_eval_begin(logs)
        self.eval_metrics = ["loss", "ppl", "nll"]

    def on_eval_batch_end(self, step, logs=None):
        super(TrainCallback, self).on_eval_batch_end(step, logs)


class LSTMEncoder(nn.Layer):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 init_scale=0.1,
                 enc_dropout=0.):
        super(LSTMEncoder, self).__init__()
        self.src_embedder = nn.Embedding(
            vocab_size,
            embed_dim,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=enc_dropout)
        if enc_dropout > 0.0:
            self.dropout = nn.Dropout(enc_dropout)
        else:
            self.dropout = None

    def forward(self, src, src_length):
        src_emb = self.src_embedder(src)

        if self.dropout:
            src_emb = self.dropout(src_emb)
        enc_output, enc_final_state = self.lstm(src_emb,
                                                sequence_length=src_length)
        if self.dropout:
            enc_output = self.dropout(enc_output)

        enc_final_state = [
            [h, c] for h, c in zip(enc_final_state[0], enc_final_state[1])
        ]
        return enc_output, enc_final_state


class LSTMDecoderCell(nn.Layer):

    def __init__(self,
                 num_layers,
                 embed_dim,
                 hidden_size,
                 latent_size,
                 dropout=None):
        super(LSTMDecoderCell, self).__init__()
        self.dropout = dropout
        self.lstm_cells = nn.LayerList([
            nn.LSTMCell(input_size=embed_dim + latent_size,
                        hidden_size=hidden_size) for i in range(num_layers)
        ])

    def forward(self, step_input, lstm_states, latent_z):
        new_lstm_states = []
        step_input = paddle.concat([step_input, latent_z], 1)
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_lstm_state = lstm_cell(step_input, lstm_states[i])
            if self.dropout:
                step_input = self.dropout(out)
            else:
                step_input = out
            new_lstm_states.append(new_lstm_state)
        if self.dropout:
            step_input = self.dropout(step_input)
        out = step_input
        return out, new_lstm_states


class LSTMDecoder(nn.Layer):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 latent_size,
                 num_layers,
                 init_scale=0.1,
                 dec_dropout=0.):
        super(LSTMDecoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.trg_embedder = nn.Embedding(
            vocab_size,
            embed_dim,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

        self.output_fc = nn.Linear(
            hidden_size,
            vocab_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

        if dec_dropout > 0.0:
            self.dropout = nn.Dropout(dec_dropout)
        else:
            self.dropout = None

        self.lstm = nn.RNN(
            LSTMDecoderCell(self.num_layers, self.embed_dim, self.hidden_size,
                            self.latent_size, self.dropout))

    def forward(self, trg, dec_initial_states, latent_z):
        trg_emb = self.trg_embedder(trg)
        if self.dropout:
            trg_emb = self.dropout(trg_emb)
        lstm_output, _ = self.lstm(inputs=trg_emb,
                                   initial_states=dec_initial_states,
                                   latent_z=latent_z)
        dec_output = self.output_fc(lstm_output)
        return dec_output


class VAESeq2SeqModel(nn.Layer):

    def __init__(self,
                 embed_dim,
                 hidden_size,
                 latent_size,
                 vocab_size,
                 num_layers=1,
                 init_scale=0.1,
                 PAD_ID=0,
                 enc_dropout=0.,
                 dec_dropout=0.):
        super(VAESeq2SeqModel, self).__init__()
        self.PAD_ID = PAD_ID
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = LSTMEncoder(vocab_size, embed_dim, hidden_size,
                                   num_layers, init_scale, enc_dropout)
        self.decoder = LSTMDecoder(vocab_size, embed_dim, hidden_size,
                                   latent_size, num_layers, init_scale,
                                   dec_dropout)
        self.distributed_fc = nn.Linear(
            hidden_size * 2,
            latent_size * 2,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))
        self.fc = nn.Linear(
            latent_size,
            2 * hidden_size * num_layers,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

    def sampling(self, z_mean, z_log_var):
        """
        Reparameterization trick 
        """
        # By default, random_normal has mean=0 and std=1.0
        epsilon = paddle.normal(shape=(z_mean.shape[0], self.latent_size))
        epsilon.stop_gradient = True
        return z_mean + paddle.exp(0.5 * z_log_var) * epsilon

    def build_distribution(self, enc_final_state=None):
        enc_hidden = [
            paddle.concat(state, axis=-1) for state in enc_final_state
        ]

        enc_hidden = paddle.concat(enc_hidden, axis=-1)
        z_mean_log_var = self.distributed_fc(enc_hidden)
        z_mean, z_log_var = paddle.split(z_mean_log_var, 2, -1)
        return z_mean, z_log_var

    def calc_kl_dvg(self, means, logvars):
        """
        Compute the KL divergence between Gaussian distribution
        """
        kl_cost = -0.5 * (logvars - paddle.square(means) - paddle.exp(logvars) +
                          1.0)
        kl_cost = paddle.mean(kl_cost, 0)

        return paddle.sum(kl_cost)

    def forward(self, src, src_length, trg, trg_length):
        # Encoder
        _, enc_final_state = self.encoder(src, src_length)

        # Build distribution
        z_mean, z_log_var = self.build_distribution(enc_final_state)

        # Decoder
        latent_z = self.sampling(z_mean, z_log_var)

        dec_first_hidden_cell = self.fc(latent_z)
        dec_first_hidden, dec_first_cell = paddle.split(dec_first_hidden_cell,
                                                        2,
                                                        axis=-1)
        if self.num_layers > 1:
            dec_first_hidden = paddle.split(dec_first_hidden, self.num_layers)
            dec_first_cell = paddle.split(dec_first_cell, self.num_layers)
        else:
            dec_first_hidden = [dec_first_hidden]
            dec_first_cell = [dec_first_cell]
        dec_initial_states = [[h, c]
                              for h, c in zip(dec_first_hidden, dec_first_cell)]

        dec_output = self.decoder(trg, dec_initial_states, latent_z)

        kl_loss = self.calc_kl_dvg(z_mean, z_log_var)
        trg_mask = (self.PAD_ID != trg).astype(paddle.get_default_dtype())
        return kl_loss, dec_output, trg_mask


class VAESeq2SeqInferModel(VAESeq2SeqModel):

    def __init__(self,
                 embed_dim,
                 hidden_size,
                 latent_size,
                 vocab_size,
                 start_token=1,
                 end_token=2,
                 beam_size=1,
                 max_out_len=100):
        self.start_token = start_token
        self.end_token = end_token
        self.beam_size = beam_size
        self.max_out_len = max_out_len
        super(VAESeq2SeqInferModel, self).__init__(embed_dim, hidden_size,
                                                   latent_size, vocab_size)

    def forward(self, trg):
        # Encoder
        latent_z = paddle.normal(shape=(trg.shape[0], self.latent_size))
        dec_first_hidden_cell = self.fc(latent_z)
        dec_first_hidden, dec_first_cell = paddle.split(dec_first_hidden_cell,
                                                        2,
                                                        axis=-1)
        if self.num_layers > 1:
            dec_first_hidden = paddle.split(dec_first_hidden, self.num_layers)
            dec_first_cell = paddle.split(dec_first_cell, self.num_layers)
        else:
            dec_first_hidden = [dec_first_hidden]
            dec_first_cell = [dec_first_cell]
        dec_initial_states = [[h, c]
                              for h, c in zip(dec_first_hidden, dec_first_cell)]

        output_fc = lambda x: F.one_hot(paddle.multinomial(
            F.softmax(paddle.squeeze(self.decoder.output_fc(x), [1]))),
                                        num_classes=self.vocab_size)

        latent_z = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            latent_z, self.beam_size)

        decoder = nn.BeamSearchDecoder(cell=self.decoder.lstm.cell,
                                       start_token=self.start_token,
                                       end_token=self.end_token,
                                       beam_size=self.beam_size,
                                       embedding_fn=self.decoder.trg_embedder,
                                       output_fn=output_fc)

        outputs, _ = nn.dynamic_decode(decoder,
                                       inits=dec_initial_states,
                                       max_step_num=self.max_out_len,
                                       latent_z=latent_z)
        return outputs
