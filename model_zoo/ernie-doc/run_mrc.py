# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import collections
from collections import namedtuple, defaultdict

import os
import random
from functools import partial
import time

import numpy as np
import paddle
import paddle.nn as nn
from paddle.optimizer import AdamW
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieDocModel
from paddlenlp.transformers import ErnieDocForQuestionAnswering
from paddlenlp.transformers import ErnieDocTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset
from paddlenlp.ops.optimizer import layerwise_lr_decay

from data import MRCIterator
from metrics import compute_qa_predictions, EM_AND_F1

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--model_name_or_path", type=str, default="ernie-doc-base-zh", help="Pretraining model name or path")
parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--learning_rate", type=float, default=2.75e-4, help="Learning rate used to train.")
parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--epochs", type=int, default=5, help="Number of epoches for training.")
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Select cpu, gpu devices to train model.")
parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
parser.add_argument("--memory_length", type=int, default=128, help="Length of the retained previous heads.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--layerwise_decay", default=0.8, type=float, help="Layerwise decay ratio")
parser.add_argument("--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", default=100, type=int, help="Max answer length.")
parser.add_argument("--do_lower_case", action='store_false', help="Whether to lower case the input text. Should be True for uncased models and False for cased models.")
parser.add_argument("--verbose", action='store_true', help="Whether to output verbose log.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout ratio of ernie_doc")
parser.add_argument("--dataset", default="dureader_robust", type=str, choices=["dureader_robust", "cmrc2018", "drcd"], help="The avaliable Q&A dataset")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

# yapf: enable
args = parser.parse_args()

# eval_dataset, test_dataset,
DATASET_INFO = {
    "dureader_robust": ["dev", "dev", ErnieDocTokenizer],
    "cmrc2018": ["dev", "dev", ErnieDocTokenizer],
    "drcd": ["dev", "test", ErnieDocTokenizer],
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def init_memory(batch_size, memory_length, d_model, n_layers):
    return [
        paddle.zeros([batch_size, memory_length, d_model], dtype="float32")
        for _ in range(n_layers)
    ]


class CrossEntropyLossForQA(paddle.nn.Layer):

    def __init__(self):
        super(CrossEntropyLossForQA, self).__init__()
        self.criterion = paddle.nn.CrossEntropyLoss()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label

        start_loss = self.criterion(start_logits, start_position)
        end_loss = self.criterion(end_logits, end_position)
        loss = (start_loss + end_loss) / 2
        return loss


@paddle.no_grad()
def evaluate(args, model, criterion, metric, data_loader, memories0, tokenizer):
    RawResult = namedtuple("RawResult",
                           ["unique_id", "start_logits", "end_logits"])
    model.eval()
    all_results = []

    tic_start = time.time()
    tic_eval = time.time()
    memories = list(memories0)

    # Collect result
    logger.info("The example number of eval_dataloader: {}".format(
        len(data_loader._batch_reader.features)))
    for step, batch in enumerate(data_loader, start=1):
        input_ids, position_ids, token_type_ids, attn_mask, start_position, \
            end_position, qids, gather_idx, need_cal_loss = batch

        start_logits, end_logits, memories = model(input_ids, memories,
                                                   token_type_ids, position_ids,
                                                   attn_mask)

        start_logits, end_logits, qids = list(
            map(lambda x: paddle.gather(x, gather_idx),
                [start_logits, end_logits, qids]))
        np_qids = qids.numpy()
        np_start_logits = start_logits.numpy()
        np_end_logits = end_logits.numpy()

        if int(need_cal_loss.numpy()) == 1:
            for idx in range(qids.shape[0]):
                if len(all_results) % 1000 == 0 and len(all_results):
                    logger.info("Processing example: %d" % len(all_results))
                    logger.info('time per 1000: {} s'.format(time.time() -
                                                             tic_eval))
                    tic_eval = time.time()

                qid_each = int(np_qids[idx])
                start_logits_each = [
                    float(x) for x in np_start_logits[idx].flat
                ]
                end_logits_each = [float(x) for x in np_end_logits[idx].flat]
                all_results.append(
                    RawResult(unique_id=qid_each,
                              start_logits=start_logits_each,
                              end_logits=end_logits_each))

    # Compute_predictions
    all_predictions_eval, all_nbest_eval = compute_qa_predictions(
        data_loader._batch_reader.examples, data_loader._batch_reader.features,
        all_results, args.n_best_size, args.max_answer_length,
        args.do_lower_case, tokenizer, args.verbose)

    EM, F1, AVG, TOTAL = metric(all_predictions_eval,
                                data_loader._batch_reader.dataset)

    logger.info("EM: {}, F1: {}, AVG: {}, TOTAL: {}, TIME: {}".format(
        EM, F1, AVG, TOTAL,
        time.time() - tic_start))
    model.train()
    return EM, F1, AVG


def do_train(args):
    set_seed(args)

    DEV, TEST, TOKENIZER_CLASS = DATASET_INFO[args.dataset]
    tokenizer = TOKENIZER_CLASS.from_pretrained(args.model_name_or_path)

    train_ds, eval_ds = load_dataset(args.dataset, splits=['train', DEV])
    if DEV == TEST:
        test_ds = eval_ds
    else:
        test_ds = load_dataset(args.dataset, splits=[TEST])

    paddle.set_device(args.device)
    trainer_num = paddle.distributed.get_world_size()
    if trainer_num > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    if rank == 0:
        if os.path.exists(args.model_name_or_path):
            logger.info("init checkpoint from %s" % args.model_name_or_path)

    model = ErnieDocForQuestionAnswering.from_pretrained(
        args.model_name_or_path, dropout=args.dropout)
    model_config = model.ernie_doc.config
    if trainer_num > 1:
        model = paddle.DataParallel(model)

    train_ds_iter = MRCIterator(train_ds,
                                args.batch_size,
                                tokenizer,
                                trainer_num,
                                trainer_id=rank,
                                memory_len=model_config["memory_len"],
                                max_seq_length=args.max_seq_length,
                                random_seed=args.seed)

    eval_ds_iter = MRCIterator(eval_ds,
                               args.batch_size,
                               tokenizer,
                               trainer_num,
                               trainer_id=rank,
                               memory_len=model_config["memory_len"],
                               max_seq_length=args.max_seq_length,
                               mode="eval",
                               random_seed=args.seed)

    test_ds_iter = MRCIterator(test_ds,
                               args.batch_size,
                               tokenizer,
                               trainer_num,
                               trainer_id=rank,
                               memory_len=model_config["memory_len"],
                               max_seq_length=args.max_seq_length,
                               mode="test",
                               random_seed=args.seed)

    train_dataloader = paddle.io.DataLoader.from_generator(capacity=70,
                                                           return_list=True)
    train_dataloader.set_batch_generator(train_ds_iter, paddle.get_device())

    eval_dataloader = paddle.io.DataLoader.from_generator(capacity=70,
                                                          return_list=True)
    eval_dataloader.set_batch_generator(eval_ds_iter, paddle.get_device())

    test_dataloader = paddle.io.DataLoader.from_generator(capacity=70,
                                                          return_list=True)
    test_dataloader.set_batch_generator(test_ds_iter, paddle.get_device())

    num_training_examples = train_ds_iter.get_num_examples()
    num_training_steps = args.epochs * num_training_examples // args.batch_size // trainer_num
    logger.info("Device count: %d, trainer_id: %d" % (trainer_num, rank))
    logger.info("Num train examples: %d" % num_training_examples)
    logger.info("Max train steps: %d" % num_training_steps)
    logger.info("Num warmup steps: %d" %
                int(num_training_steps * args.warmup_proportion))

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    # Construct dict
    name_dict = dict()
    for n, p in model.named_parameters():
        name_dict[p.name] = n

    simple_lr_setting = partial(layerwise_lr_decay, args.layerwise_decay,
                                name_dict, model_config["num_hidden_layers"])

    optimizer = AdamW(learning_rate=lr_scheduler,
                      parameters=model.parameters(),
                      weight_decay=args.weight_decay,
                      apply_decay_param_fun=lambda x: x in decay_params,
                      lr_ratio=simple_lr_setting)

    global_steps = 0
    create_memory = partial(init_memory, args.batch_size, args.memory_length,
                            model_config["hidden_size"],
                            model_config["num_hidden_layers"])

    criterion = CrossEntropyLossForQA()

    memories = create_memory()
    tic_train = time.time()
    best_avg_metric = -1
    stop_training = False
    for epoch in range(args.epochs):
        train_ds_iter.shuffle_sample()
        train_dataloader.set_batch_generator(train_ds_iter, paddle.get_device())
        for step, batch in enumerate(train_dataloader, start=1):
            global_steps += 1
            input_ids, position_ids, token_type_ids, attn_mask, start_position, \
                end_position, qids, gather_idx, need_cal_loss = batch
            start_logits, end_logits, memories = model(input_ids, memories,
                                                       token_type_ids,
                                                       position_ids, attn_mask)

            start_logits, end_logits, qids, start_position, end_position = list(
                map(lambda x: paddle.gather(x, gather_idx), [
                    start_logits, end_logits, qids, start_position, end_position
                ]))
            loss = criterion([start_logits, end_logits],
                             [start_position, end_position]) * need_cal_loss

            mean_loss = loss.mean()
            mean_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_steps % args.logging_steps == 0:
                logger.info(
                    "train: global step %d, epoch: %d, loss: %f, lr: %f, speed: %.2f step/s"
                    % (global_steps, epoch, mean_loss, lr_scheduler.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_steps % args.save_steps == 0:
                # Evaluate
                logger.info("Eval:")
                EM, F1, AVG = evaluate(args, model, criterion,
                                       EM_AND_F1(), eval_dataloader,
                                       create_memory(), tokenizer)
                if rank == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "model_%d" % (global_steps))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    if best_avg_metric < AVG:
                        output_dir = os.path.join(args.output_dir, "best_model")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

            if args.max_steps > 0 and global_steps >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            break
    logger.info("Test:")
    evaluate(args, model, criterion, EM_AND_F1(), test_dataloader,
             create_memory(), tokenizer)
    if rank == 0:
        output_dir = os.path.join(args.output_dir, "model_%d" % (global_steps))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model._layers if isinstance(
            model, paddle.DataParallel) else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    do_train(args)
