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
import os
import random
import time
import math
from functools import partial
import distutils.util
import numpy as np
import paddle
from paddle.io import DataLoader

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import RobertaForMaskedLM, RobertaTokenizer, RobertaModel
from paddlenlp.data import Stack, Tuple, Pad, Dict
import copy
from tqdm import tqdm
from utils import DataCollatorMLM

parser = argparse.ArgumentParser()
IGNORE = -100

# yapf: disable
parser.add_argument("--model_name_or_path", default='roberta-en-base', type=str, required=False, help="Path to pre-trained model")
parser.add_argument("--input_file", default='wiki', type=str, required=False, help="The input directory where the model predictions and checkpoints will be written.")
parser.add_argument("--output_dir", default='ckp/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every X updates steps.")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument("--scale_loss", type=float, default=2**15, help="The value of scale_loss for fp16.")
parser.add_argument("--amp", type=distutils.util.strtobool,default=True, help="use mix precision.")

roberta_arch = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 514,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": 50265,
            "layer_norm_eps": 1e-05,
            "pad_token_id": 1,
            "cls_token_id": 0
        }

def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(seed)

def do_train(args):
    paddle.set_device(args.device)
    set_seed(args.seed)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Load model and train from scratch
    # model = RobertaForMaskedLM(
    #     RobertaModel(**RobertaForMaskedLM.pretrained_init_configuration[
    #         args.model_name_or_path]))
    model = RobertaForMaskedLM(RobertaModel(**roberta_arch))
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    ignore_label = IGNORE
    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    # Load wikipedia dataset via Hugging face datasets
    # TO DO: paddle datasets
    import datasets
    tokenized_datasets = datasets.load_from_disk(args.input_file)
    train_ds = tokenized_datasets["train"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Prepare data for training
    collator_func = DataCollatorMLM(tokenizer=tokenizer) # data collator
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_data_loader = DataLoader(
        dataset=train_ds,
        collate_fn=collator_func,
        num_workers=0,
        batch_sampler=train_batch_sampler,
        return_list=True)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    num_training_steps = args.max_steps if args.max_steps > 0 else len(train_data_loader) * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
    if args.amp: #mixed precision (fp16)
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    # Start training
    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            input_ids, _, labels = batch
            with paddle.amp.auto_cast(
                        args.amp,
                        #custom_white_list=["layer_norm", "softmax", "gelu"]
                        ):
                logits = model(input_ids=input_ids)
                loss = loss_fct(logits, labels)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
                # print(args.amp, args.learning_rate, args.weight_decay)
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % args.logging_steps == 0:

                print(
                    "global step %d/%d, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, loss, optimizer.get_lr(),
                        args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                                "paddle_%d" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    do_train(args)
