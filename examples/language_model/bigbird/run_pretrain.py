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
import itertools
import logging
import os
import random
import time
import h5py
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, Dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BigBirdForPretraining, BigBirdModel, BigBirdPretrainingCriterion
from paddlenlp.transformers import BigBirdTokenizer

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {"bigbird": (BigBirdForPretraining, BigBirdTokenizer), }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default="bigbird-base-uncased",
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--max_predictions_per_seq",
        default=80,
        type=int,
        help="The maximum total of masked tokens in input sequence")

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epoches for training.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default='~/',
        help="vocab file used to tokenize text")
    parser.add_argument(
        "--vocab_model_file",
        type=str,
        default='sentencepiece_gpt2.model',
        help="vocab model file used to tokenize text")
    parser.add_argument(
        "--max_encoder_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after SentencePiece tokenization."
    )
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="")
    parser.add_argument("--hidden_size", type=int, default=768, help="")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=3072)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--normalize_before", type=bool, default=False)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--num_rand_blocks", type=int, default=3)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_position_embeddings", type=int, default=4096)
    parser.add_argument("--type_vocab_size", type=int, default=2)
    parser.add_argument("--data_file", type=str, default="train.csv")
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class PretrainingDataset(Dataset):
    def __init__(self,
                 input_file,
                 tokenizer,
                 max_encoder_length=512,
                 max_predictions_per_seq=75,
                 masked_lm_prob=0.15,
                 pad_val=0,
                 cls_val=65,
                 sep_val=66,
                 mask_val=67,
                 mask_prob=0.8,
                 random_prob=0.1):
        self.tokenizer = tokenizer
        self.max_encoder_length = max_encoder_length
        self.max_predictions_per_seq = max_predictions_per_seq
        self.pad_val = pad_val
        input_file = open(input_file, "r")
        self.lines = input_file.readlines()

        self.vocab_size = tokenizer.vocab_size
        self.word_start_subtoken = np.array([
            tokenizer.vocab.idx_to_token[i][0] == "▁"
            for i in range(self.vocab_size)
        ])
        self.masked_lm_prob = masked_lm_prob
        self.cls_val = cls_val
        self.sep_val = sep_val
        self.mask_val = mask_val
        self.mask_prob = mask_prob
        self.random_prob = random_prob

    def _pretrain_masking(self, subtokens):
        end_pos = self.max_encoder_length - 2 + np.random.randint(
            max(1, len(subtokens) - self.max_encoder_length - 2))
        start_pos = max(0, end_pos - self.max_encoder_length + 2)
        subtokens = np.array(subtokens[start_pos:end_pos])

        word_begin_mark = self.word_start_subtoken[subtokens]
        word_begins_pos = np.flatnonzero(word_begin_mark).astype(np.int32)
        if word_begins_pos.size == 0:
            # if no word boundary present, we do not do whole word masking
            # and we fall back to random masking.
            word_begins_pos = np.arange(len(subtokens), dtype=np.int32)
            word_begin_mark = np.logical_not(word_begin_mark)
        correct_start_pos = word_begins_pos[0]
        subtokens = subtokens[correct_start_pos:]
        word_begin_mark = word_begin_mark[correct_start_pos:]
        word_begins_pos = word_begins_pos - correct_start_pos
        num_tokens = len(subtokens)

        words = np.split(
            np.arange(
                num_tokens, dtype=np.int32), word_begins_pos)[1:]
        assert len(words) == len(word_begins_pos)

        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(len(word_begins_pos) * self.masked_lm_prob))))
        masked_lm_positions = np.concatenate(
            np.random.choice(
                np.array(
                    [[]] + words, dtype=np.object)[1:],
                num_to_predict,
                replace=False),
            0)

        if len(masked_lm_positions) > self.max_predictions_per_seq:
            masked_lm_positions = masked_lm_positions[:self.
                                                      max_predictions_per_seq +
                                                      1]
            # however last word can cross word boundaries, remove crossing words
            truncate_masking_at = np.flatnonzero(word_begin_mark[
                masked_lm_positions])[-1]
            masked_lm_positions = masked_lm_positions[:truncate_masking_at]

        masked_lm_positions = np.sort(masked_lm_positions)
        masked_lm_ids = subtokens[masked_lm_positions]

        randomness = np.random.rand(len(masked_lm_positions))

        mask_index = masked_lm_positions[randomness < self.mask_prob]
        random_index = masked_lm_positions[randomness > (1 - self.random_prob)]

        subtokens[mask_index] = self.mask_val  # id of masked token
        subtokens[random_index] = np.random.randint(  # ignore special tokens
            101,
            self.vocab_size,
            len(random_index),
            dtype=np.int32)

        subtokens = np.concatenate([
            np.array(
                [self.cls_val], dtype=np.int32), subtokens, np.array(
                    [self.sep_val], dtype=np.int32)
        ])

        # pad everything to correct shape
        pad_inp = self.max_encoder_length - num_tokens - 2
        subtokens = np.pad(subtokens, [0, pad_inp], "constant")

        pad_out = self.max_predictions_per_seq - len(masked_lm_positions)
        masked_lm_weights = np.pad(np.ones_like(
            masked_lm_positions, dtype=np.float32), [0, pad_out],
                                   "constant")
        masked_lm_positions = np.pad(masked_lm_positions + 1, [0, pad_out],
                                     "constant")
        masked_lm_ids = np.pad(masked_lm_ids, [0, pad_out], "constant")

        return subtokens, masked_lm_positions, masked_lm_ids, masked_lm_weights

    def __getitem__(self, index):
        # [input_ids, label]
        line = self.lines[index].rstrip()
        # numpy_mask
        subtokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer(line))

        subtokens, masked_lm_positions, masked_lm_ids, masked_lm_weights = \
                self._pretrain_masking(subtokens)
        return [
            subtokens, np.zeros_like(subtokens), masked_lm_positions,
            masked_lm_ids, masked_lm_weights, np.zeros(
                [1], dtype="int64")
        ]

    def __len__(self):
        return len(self.lines)


def create_dataloader(input_file, tokenizer, worker_init, batch_size):
    pretrain_dataset = PretrainingDataset(input_file, tokenizer)
    train_batch_sampler = paddle.io.BatchSampler(
        pretrain_dataset, batch_size=batch_size, shuffle=True)
    # TODO(zhoushunjie): 后续考虑优化masked_lm_position
    # make masked_lm_positions can be gathered
    # def _collate_data(data, stack_fn=Stack()):
    #     # data: input_ids, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels
    #     #num_fields = len(data[0].keys())
    #     out = {}

    #     for i in data[0].keys():
    #         out[i] = stack_fn([x[i] for x in data])
    #     # batch_size, seq_length = out[0].shape
    #     # # Organize as a 1D tensor for gather or use gather_nd
    #     # mask_token_num = 0
    #     # for i, x in enumerate(data):
    #     #     for j, pos in enumerate(x[2]):
    #     #         out[2][mask_token_num] = i * seq_length + pos
    #     #         mask_token_num += 1
    #     # out[2] = 
    #     print(out,flush=True)
    #     return out

    dataloader = DataLoader(
        dataset=pretrain_dataset,
        batch_sampler=train_batch_sampler,
        # collate_fn=_collate_data,
        num_workers=0,
        worker_init_fn=worker_init,
        return_list=True)
    return dataloader


def do_train(args):

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())

    # get dataloader
    tokenizer = BigBirdTokenizer.from_pretrained(args.model_name_or_path)
    dataloader = create_dataloader(args.data_file, tokenizer, worker_init,
                                   args.batch_size)

    # define model
    model = BigBirdForPretraining.from_pretrained(args.model_name_or_path)

    # define metric

    # define optimizer

    # training


if __name__ == "__main__":
    args = parse_args()
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
