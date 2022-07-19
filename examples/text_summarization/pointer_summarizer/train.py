import os
import sys

import config
from data import Batcher, Vocab
from utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from model import Model

import numpy as np
import time
import argparse

import paddle
import paddle.nn as nn
from paddle.optimizer import Adagrad, Adam, SGD

# Flush out immediately
sys.stdout.flush()


class Trainer(object):

    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path,
                               self.vocab,
                               mode='train',
                               batch_size=config.batch_size,
                               single_pass=False)

        train_dir = os.path.join(config.log_root,
                                 'train_%d' % (int(time.time())))

        if not os.path.exists(config.log_root):
            os.mkdir(config.log_root)

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'encoder': self.model.encoder.state_dict(),
            'decoder': self.model.decoder.state_dict(),
            'reduce_state': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        model_save_dir = os.path.join(
            self.model_dir, 'model_%06d_%.8f' % (iter, running_avg_loss))
        for k in state:
            model_save_path = os.path.join(model_save_dir, '%s.params' % k)
            paddle.save(state[k], model_save_path)
        return model_save_dir

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        assert len(params) == 31
        self.optimizer = Adagrad(
            parameters=params,
            learning_rate=initial_lr,
            initial_accumulator_value=config.adagrad_init_acc,
            epsilon=1.0e-10,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(
                clip_norm=config.max_grad_norm))

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            start_iter = int(model_file_path.split('_')[-2])
            start_loss = float(
                model_file_path.split('_')[-1].replace(os.sep, ''))

            if not config.is_coverage:
                self.optimizer.set_state_dict(
                    paddle.load(
                        os.path.join(model_file_path, 'optimizer.params')))

        return start_iter, start_loss

    def train_one_batch(self, batch, iter):

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch)

        self.optimizer.clear_gradients()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(
            enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]

            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
                self.model.decoder(y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                                   c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, di)

            target = target_batch[:, di]
            add_index = paddle.arange(0, target.shape[0])
            new_index = paddle.stack([add_index, target], axis=1)
            gold_probs = paddle.gather_nd(final_dist, new_index).squeeze()
            step_loss = -paddle.log(gold_probs + config.eps)

            if config.is_coverage:
                step_coverage_loss = paddle.sum(
                    paddle.minimum(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = paddle.sum(paddle.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = paddle.mean(batch_avg_loss)

        loss.backward()
        self.optimizer.minimize(loss)

        return loss.numpy()[0]

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch, iter)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss,
                                                     iter)
            iter += 1
            print(
                'global step %d/%d, step loss: %.8f, running avg loss: %.8f, speed: %.2f step/s'
                % (iter, n_iters, loss, running_avg_loss, 1.0 /
                   (time.time() - start)))
            start = time.time()
            if iter % 5000 == 0 or iter == 1000:
                model_save_dir = self.save_model(running_avg_loss, iter)
                print(
                    'Saved model for iter %d with running avg loss %.8f to directory: %s'
                    % (iter, running_avg_loss, model_save_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override config.max_iterations.",
    )
    args = parser.parse_args()

    train_processor = Trainer()
    if args.max_steps > 0:
        train_processor.trainIters(args.max_steps, args.model_file_path)
    else:
        train_processor.trainIters(config.max_iterations, args.model_file_path)
