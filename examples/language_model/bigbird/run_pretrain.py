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
        default=None,
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


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args,
                               worker_init):
    train_data = PretrainingDataset(
        input_file=input_file, max_pred_length=max_pred_length)
    # files have been sharded, no need to dispatch again
    train_batch_sampler = paddle.io.BatchSampler(
        train_data, batch_size=args.batch_size, shuffle=True)

    # DataLoader cannot be pickled because of its place.
    # If it can be pickled, use global function instead of lambda and use
    # ProcessPoolExecutor instead of ThreadPoolExecutor to prefetch.
    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        # input_ids, segment_ids, input_mask, masked_lm_positions,
        # masked_lm_labels, next_sentence_labels, mask_token_num
        for i in (0, 1, 2, 5):
            out[i] = stack_fn([x[i] for x in data])
        batch_size, seq_length = out[0].shape
        size = num_mask = sum(len(x[3]) for x in data)
        # Padding for divisibility by 8 for fp16 or int8 usage
        if size % 8 != 0:
            size += 8 - (size % 8)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        out[3] = np.full(size, 0, dtype=np.int64)
        # masked_lm_labels
        out[4] = np.full([size, 1], -1, dtype=np.int64)
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[3]):
                out[3][mask_token_num] = i * seq_length + pos
                out[4][mask_token_num] = x[4][j]
                mask_token_num += 1
        # mask_token_num
        out.append(np.asarray([mask_token_num], dtype=np.float32))
        return out

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_sampler=train_batch_sampler,
        collate_fn=_collate_data,
        num_workers=0,
        worker_init_fn=worker_init,
        return_list=True)
    return train_data_loader, input_file


class PretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            input[index].astype(np.int64)
            if indice < 5 else np.asarray(input[index].astype(np.int64))
            for indice, input in enumerate(self.inputs)
        ]
        # TODO: whether to use reversed mask by changing 1s and 0s to be
        input_mask = (1 - np.reshape(
            input_mask.astype(np.float32), [1, 1, input_mask.shape[0]])) * -1e9

        index = self.max_pred_length
        # store number of  masked tokens in index
        # outputs of torch.nonzero diff with that of numpy.nonzero by zip
        padded_mask_indices = (masked_lm_positions == 0).nonzero()[0]
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
            mask_token_num = index
        else:
            index = 0
            mask_token_num = 0
        # masked_lm_labels = np.full(input_ids.shape, -1, dtype=np.int64)
        # masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        masked_lm_labels = masked_lm_ids[:index]
        masked_lm_positions = masked_lm_positions[:index]
        # softmax_with_cross_entropy enforce last dim size equal 1
        masked_lm_labels = np.expand_dims(masked_lm_labels, axis=-1)
        next_sentence_labels = np.expand_dims(next_sentence_labels, axis=-1)

        return [
            input_ids, segment_ids, input_mask, masked_lm_positions,
            masked_lm_labels, next_sentence_labels
        ]


def do_train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    model = BigBirdForPretraining(
        BigBirdModel(**model_class.pretrained_init_configuration[
            args.model_name_or_path]))
    criterion = BigBirdPretrainingCriterion(
        getattr(model, BigBirdForPretraining.base_model_prefix).config[
            "vocab_size"])
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(args.learning_rate,
                                                       args.max_steps, 0)
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        lr_scheduler,
        args.warmup_steps,
        start_lr=0.0,
        end_lr=args.learning_rate)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and "npz_"
                not in str(f))
        ]
        files.sort()
        num_files = len(files)
        for f_id in range(num_files):
            data_file = files[f_id]
            train_data_loader = create_pretraining_dataset(
                args, data_file, worker_init, worker_index, eod_id=eod_id)
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                (input_ids, segment_ids, input_mask, masked_lm_positions,
                 masked_lm_labels, next_sentence_labels,
                 masked_lm_scale) = batch
                prediction_scores, seq_relationship_score = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    masked_positions=masked_lm_positions)
                loss = criterion(prediction_scores, seq_relationship_score,
                                 masked_lm_labels, next_sentence_labels,
                                 masked_lm_scale)
                if global_step % args.logging_steps == 0:
                    if (not args.n_gpu > 1
                        ) or paddle.distributed.get_rank() == 0:
                        logger.info(
                            "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                            % (global_step, epoch, step, loss,
                               args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_gradients()
                if global_step % args.save_steps == 0:
                    if (not args.n_gpu > 1
                        ) or paddle.distributed.get_rank() == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        paddle.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "model_state.pdopt"))
                if global_step >= args.max_steps:
                    del train_data_loader
                    return

            del train_data_loader
            train_data_loader, data_file = dataset_future.result(timeout=None)


if __name__ == "__main__":
    args = parse_args()
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
