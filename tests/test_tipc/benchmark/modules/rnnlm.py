import os
import sys

import paddle
import paddle.nn as nn

from paddlenlp.utils import profiler
from paddlenlp.utils.log import logger
from paddlenlp.metrics import Perplexity

from .model_base import BenchmarkBase

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                     os.pardir, "examples", "language_model")))
from rnnlm.reader import create_data_loader
from rnnlm.model import RnnLm, CrossEntropyLossForLm, UpdateModel


class AddProfiler(paddle.callbacks.Callback):

    def on_batch_end(self, mode, step=None, logs=None):
        if mode == 'train':
            profiler.add_profiler_step(self.profiler_options)


class RNNLMBenchmark(BenchmarkBase):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument('--hidden_size',
                            type=int,
                            default=650,
                            help='hidden_size')
        parser.add_argument('--num_steps',
                            type=int,
                            default=35,
                            help='num steps')
        parser.add_argument('--num_layers',
                            type=int,
                            default=2,
                            help='num_layers')
        parser.add_argument('--dropout',
                            type=float,
                            default=0.5,
                            help='dropout')
        parser.add_argument('--init_scale',
                            type=float,
                            default=0.05,
                            help='init_scale')
        parser.add_argument('--use_hapi',
                            action="store_false",
                            help="Whether to use hapi to run. ")

    def create_data_loader(self, args, **kwargs):
        train_loader, valid_loader, test_loader, self.vocab_size = create_data_loader(
            batch_size=args.batch_size, num_steps=args.num_steps)

        self.num_batch = len(train_loader)

        return train_loader, valid_loader

    def build_model(self, args, **kwargs):
        network = RnnLm(vocab_size=self.vocab_size,
                        hidden_size=args.hidden_size,
                        batch_size=args.batch_size,
                        num_layers=args.num_layers,
                        init_scale=args.init_scale,
                        dropout=args.dropout)

        self.cross_entropy = CrossEntropyLossForLm()

        model = paddle.Model(network)

        return model

    def forward(self, model, args, input_data=None, **kwargs):
        ppl_metric = Perplexity()
        callback = UpdateModel()

        scheduler = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True)

        model.prepare(optimizer=kwargs.get("optimizer"),
                      loss=self.cross_entropy,
                      metrics=ppl_metric)

        benchmark_logger = self.logger(args)

        if args.profiler_options is not None:
            profiler_callback = AddProfiler()
            profiler_callback.profiler_options = args.profiler_options
            callbacks_lists = [
                callback, scheduler, benchmark_logger, profiler_callback
            ]
        else:
            callbacks_lists = [callback, scheduler, benchmark_logger]

        model.fit(train_data=kwargs.get("train_loader"),
                  eval_data=kwargs.get("eval_loader"),
                  epochs=args.epoch,
                  shuffle=False,
                  callbacks=callbacks_lists)

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
        return paddle.callbacks.ProgBarLogger(log_freq=(self.num_batch // 10),
                                              verbose=3)
