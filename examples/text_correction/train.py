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

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import ErnieGramTokenizer
from paddlenlp.utils.log import logger

from model import ErnieGramForCSC
from data import read_train_ds, convert_example, create_dataloader

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--model_name_or_path", type=str, default="ernie-gram-zh", help="Pretraining model name or path")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--epochs", type=int, default=3, help="Number of epoches for training.")
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Select cpu, gpu devices to train model.")
parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--pinyin_vocab_file_path", type=str, default="pinyin_vocab.txt", help="pinyin vocab file path")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--ignore_label", default=-1, type=int, help="Ignore label for CrossEntropyLoss")

# yapf: enable
args = parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def do_train(args):
    set_seed(args)
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    pinyin_vocab = Vocab.load_vocabulary(
        args.pinyin_vocab_file_path, unk_token='[UNK]', pad_token='[PAD]')

    tokenizer = ErnieGramTokenizer.from_pretrained(args.model_name_or_path)
    model = ErnieGramForCSC.from_pretrained(
        args.model_name_or_path,
        pinyin_vocab_size=len(pinyin_vocab),
        pad_pinyin_id=pinyin_vocab[pinyin_vocab.pad_token])

    train_ds = load_dataset(read_train_ds, data_path='train.txt', lazy=False)

    det_loss_act = paddle.nn.CrossEntropyLoss(
        ignore_index=args.ignore_label, use_softmax=False)
    corr_loss_act = paddle.nn.CrossEntropyLoss(
        ignore_index=args.ignore_label, reduction='none')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        pinyin_vocab=pinyin_vocab,
        max_seq_length=args.max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Pad(axis=0, pad_val=pinyin_vocab.token_to_idx[pinyin_vocab.pad_token]),  # pinyin
        Pad(axis=0, dtype="int64"),  # detection label
        Pad(axis=0, dtype="int64")  # correction label
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    logger.info("Total training step: {}".format(num_training_steps))
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    global_steps = 1

    tic_train = time.time()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_data_loader):
            input_ids, token_type_ids, pinyin_ids, det_labels, corr_labels = batch
            det_error_probs, corr_logits = model(input_ids, pinyin_ids,
                                                 token_type_ids)

            # det_error_probs is producted by softmax
            det_loss = det_loss_act(det_error_probs, det_labels)

            init_corr_loss = corr_loss_act(corr_logits, corr_labels)
            corr_loss = init_corr_loss * det_error_probs.max(axis=-1)

            loss = (det_loss + corr_loss).mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_steps % args.logging_steps == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_steps, epoch, step, loss,
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_steps % args.save_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    paddle.save(
                        model.state_dict(),
                        os.path.join(args.output_dir,
                                     "model_%d.pdparams" % global_steps))
                    logger.info("Save model at {} step.".format(global_steps))
            if args.max_steps > 0 and global_steps >= args.max_steps:
                return
            global_steps += 1

            # print("batch:", batch, flush=True)
            # print("det_error_probs:", det_error_probs, flush=True)
            # print("corr_logits:", corr_logits, flush=True)


if __name__ == "__main__":
    do_train(args)
