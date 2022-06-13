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

import os
import logging
import random
import time
import numpy as np

import paddle
from paddle.io import DataLoader, Dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BigBirdForPretraining, BigBirdModel, BigBirdPretrainingCriterion
from paddlenlp.transformers import BigBirdTokenizer, LinearDecayWithWarmup, create_bigbird_rand_mask_idx_list
from paddlenlp.utils.log import logger
import args

MODEL_CLASSES = {
    "bigbird": (BigBirdForPretraining, BigBirdTokenizer),
}


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
                 max_pred_length=75):
        self.tokenizer = tokenizer
        self.max_encoder_length = max_encoder_length
        self.max_pred_length = max_pred_length
        input_file = open(input_file, "r", encoding="utf-8")
        self.lines = input_file.readlines()

        self.vocab_size = tokenizer.vocab_size

    def __getitem__(self, index):
        line = self.lines[index].rstrip()
        subtokens, masked_lm_positions, masked_lm_ids, masked_lm_weights = self.tokenizer.encode(
            line,
            max_seq_len=self.max_encoder_length,
            max_pred_len=self.max_pred_length)
        return [
            subtokens,
            np.zeros_like(subtokens), masked_lm_positions, masked_lm_ids,
            masked_lm_weights,
            np.zeros([1], dtype="int64")
        ]

    def __len__(self):
        return len(self.lines)


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


def create_dataloader(input_file, tokenizer, worker_init, batch_size,
                      max_encoder_length, max_pred_length, config):
    pretrain_dataset = PretrainingDataset(input_file, tokenizer,
                                          max_encoder_length, max_pred_length)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        pretrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # make masked_lm_positions can be gathered
    def _collate_data(data, stack_fn=Stack()):
        # Data Fields: input_ids, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels
        num_fields = len(data[0])
        out = [None] * num_fields

        for i in [0, 1, 5]:
            out[i] = stack_fn([x[i] for x in data])
        batch_size, seq_length = out[0].shape
        size = num_mask = sum(len(x[2]) for x in data)
        out[2] = np.full(size, 0, dtype=np.int32)
        # masked_lm_labels
        out[3] = np.full([size, 1], -1, dtype=np.int64)
        # masked weight
        out[4] = np.full([size], 0, dtype="float32")
        # # Organize as a 1D tensor for gather or use gather_nd
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[2]):
                out[2][mask_token_num] = i * seq_length + pos
                out[3][mask_token_num] = x[3][j]
                out[4][mask_token_num] = x[4][j]
                mask_token_num += 1
        out.append(np.asarray([mask_token_num], dtype=np.float32))
        seq_len = len(out[0][0])
        rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
            config["num_layers"], seq_len, seq_len, config["nhead"],
            config["block_size"], config["window_size"],
            config["num_global_blocks"], config["num_rand_blocks"],
            config["seed"])
        out.extend(rand_mask_idx_list)
        return out

    dataloader = DataLoader(dataset=pretrain_dataset,
                            batch_sampler=train_batch_sampler,
                            collate_fn=_collate_data,
                            worker_init_fn=worker_init,
                            return_list=True)
    return dataloader


def do_train(args):
    # Initialization for the parallel enviroment
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()

    # Set the random seed for the training process
    set_seed(args)
    worker_init = WorkerInitObj(args.seed + worker_index)

    # Get the model class and tokenizer class
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # Define the pretrain model and metric
    pretrained_models_list = list(
        model_class.pretrained_init_configuration.keys())
    if args.model_name_or_path in pretrained_models_list:
        model = BigBirdForPretraining(
            BigBirdModel(**model_class.pretrained_init_configuration[
                args.model_name_or_path]))
    else:
        model = BigBirdForPretraining.from_pretrained(args.model_name_or_path)
    # Get bigbird config for generate random attention mask
    config = getattr(model, BigBirdForPretraining.base_model_prefix).config
    criterion = BigBirdPretrainingCriterion(config["vocab_size"], args.use_nsp)
    if worker_num > 1:
        model = paddle.DataParallel(model)

    # Define learing_rate scheduler and optimizer
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, args.max_steps,
                                         args.warmup_steps)

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

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.epochs):
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        ]
        files.sort()
        num_files = len(files)
        for f_id in range(num_files):
            train_data_loader = create_dataloader(files[f_id], tokenizer,
                                                  worker_init, args.batch_size,
                                                  args.max_encoder_length,
                                                  args.max_pred_length, config)
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                (input_ids, segment_ids, masked_lm_positions, masked_lm_ids,
                 masked_lm_weights, next_sentence_labels,
                 masked_lm_scale) = batch[:7]
                rand_mask_idx_list = batch[7:]

                prediction_scores, seq_relationship_score = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    rand_mask_idx_list=rand_mask_idx_list,
                    masked_positions=masked_lm_positions)
                loss = criterion(prediction_scores, seq_relationship_score,
                                 masked_lm_ids, next_sentence_labels,
                                 masked_lm_scale, masked_lm_weights)
                if global_step % args.logging_steps == 0 and worker_index == 0:
                    logger.info(
                        "global step %d, epoch: %d, lr: %.10f, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, optimizer.get_lr(), loss,
                           args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % args.save_steps == 0:
                    if worker_index == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Need better way to get inner model of DataParallel
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


if __name__ == "__main__":
    args = args.parse_args()
    do_train(args)
