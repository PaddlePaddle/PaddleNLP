import os
import sys

import paddle

from paddlenlp.utils.log import logger
from paddlenlp.transformers.xlnet.modeling import XLNetPretrainedModel, XLNetForSequenceClassification
from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer

from .model_base import BenchmarkBase

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                     os.pardir, "examples", "language_model")))
from xlnet.run_glue import create_data_loader


class XLNetBenchmark(BenchmarkBase):

    def __init__(self):
        self.label_list = None
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument('--model_name_or_path',
                            type=str,
                            default="xlnet-base-cased",
                            help='Model name. Defaults to xlnet-base-cased. ')
        parser.add_argument('--task_name',
                            type=str,
                            default="SST-2",
                            help='Task name. Defaults to sst-2. ')
        parser.add_argument('--max_seq_length',
                            type=int,
                            default=args.max_seq_len,
                            help='Maximum sequence length. ')

    def create_data_loader(self, args, **kwargs):
        args.task_name = args.task_name.lower()
        tokenizer = XLNetTokenizer.from_pretrained(args.model_name_or_path)

        if args.task_name == "mnli":
            train_data_loader, dev_data_loader_matched, dev_data_loader_mismatched, train_ds, _, _ = create_data_loader(
                args, tokenizer)
        else:
            train_loader, dev_loader, train_ds, _ = create_data_loader(
                args, tokenizer)

        self.num_batch = len(train_loader)
        self.label_list = train_ds.label_list

        if args.task_name == "mnli":
            return train_data_loader, (dev_data_loader_matched,
                                       dev_data_loader_mismatched)
        else:
            return train_loader, dev_loader

    def build_model(self, args, **kwargs):
        num_classes = 1 if self.label_list is None else len(self.label_list)
        model = XLNetForSequenceClassification.from_pretrained(
            args.model_name_or_path, num_classes=num_classes)

        self.loss_fct = paddle.nn.loss.CrossEntropyLoss(
        ) if self.label_list else paddle.nn.loss.MSELoss()

        return model

    def forward(self, model, args, input_data=None, **kwargs):
        input_ids, token_type_ids, attention_mask, labels = input_data
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = self.loss_fct(logits, labels)

        return loss, args.batch_size

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
