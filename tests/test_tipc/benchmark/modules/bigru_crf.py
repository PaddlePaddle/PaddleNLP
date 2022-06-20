import os
import sys

import paddle

from paddlenlp.utils.log import logger

from .model_base import BenchmarkBase

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from bigru_crf.data import create_data_loader
from bigru_crf.model import BiGruCrf


class BiGruCrfBenchmark(BenchmarkBase):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument(
            "--base_lr",
            type=float,
            default=0.001,
            help="The basic learning rate that affects the entire network.")
        parser.add_argument(
            "--crf_lr",
            type=float,
            default=0.2,
            help="The learning rate ratio that affects CRF layers.")
        parser.add_argument("--emb_dim",
                            type=int,
                            default=128,
                            help="The dimension in which a word is embedded.")
        parser.add_argument("--hidden_size",
                            type=int,
                            default=128,
                            help="The number of hidden nodes in the GRU layer.")

        return parser

    def create_data_loader(self, args, **kwargs):
        self.word_vocab, self.label_vocab, train_loader, test_loader = create_data_loader(
            args)

        self.num_batch = len(train_loader)

        return train_loader, test_loader

    def build_model(self, args, **kwargs):
        model = BiGruCrf(args.emb_dim,
                         args.hidden_size,
                         len(self.word_vocab),
                         len(self.label_vocab),
                         crf_lr=args.crf_lr)

        return model

    def forward(self, model, args, input_data=None, **kwargs):
        (token_ids, length, label_ids) = input_data
        loss = model(token_ids, length, label_ids)
        avg_loss = paddle.mean(loss)

        return avg_loss, args.batch_size

    def logger(self,
               args,
               step_id=None,
               pass_id=None,
               batch_id=None,
               loss=None,
               batch_cost=None,
               reader_cost=None,
               num_samples=None,
               ips=None,
               **kwargs):
        logger.info(
            "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sequences/sec"
            % (step_id, args.epoch * self.num_batch, loss, reader_cost,
               batch_cost, num_samples, ips))
