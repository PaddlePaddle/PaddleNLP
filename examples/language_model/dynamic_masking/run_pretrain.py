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

import paddlenlp as ppnlp
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieForPretraining, ErnieTokenizer, ErnieModel
from paddlenlp.data import Stack, Tuple, Pad, Dict
import copy
from tqdm import tqdm

from modeling import ErnieForMLMPretraining
from collator import DataCollatorMLM

parser = argparse.ArgumentParser()
IGNORE = -100

# yapf: disable
parser.add_argument("--model_name_or_path", default='ernie-2.0-en', type=str, required=False, help="Path to pre-trained model")
# parser.add_argument("--input_file", default='/work/test/hf/collator/hf/sep3/TrainData_line.jsonl', type=str, required=False, help="The input directory where the model predictions and checkpoints will be written.")
parser.add_argument("--input_file", default='/work/test/twitter_hash_mask_tmp.txt', type=str, required=False, help="The input directory where the model predictions and checkpoints will be written.")

parser.add_argument("--output_dir", default='/work/test/hf/collator/pd/tmp/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every X updates steps.")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--device", default="gpu:4", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument("--scale_loss", type=float, default=2**15, help="The value of scale_loss for fp16.")
parser.add_argument("--amp", type=distutils.util.strtobool,default=False, help="use mix precision.")

def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(seed)

def _tokenize(example, tokenizer, max_len=128):
    # Tokenization fuction
    example_token = tokenizer(example, max_seq_len=max_len, 
        return_special_tokens_mask=True)
    example_token['special_tokens_mask'] = [1] + example_token['special_tokens_mask'] +[1]
    return example_token


def read_data(fileName):
    # Read the raw txt file, each line represents one sample.
    # customize the read function if necessary
    with open(fileName, 'r') as f:
        for line in f:
            yield line
        f.close()

def do_train(args):
    paddle.set_device(args.device)
    set_seed(args.seed)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    
    # Read and tokenize the data
    # Tokenizer will return dict: {'input_ids':[...], 'token_type_ids':[...], 'special_tokens_mask':[...]}
    train_ds = load_dataset(read_data, fileName=args.input_file, lazy=False)
    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)
    trans_func = partial(
        _tokenize,
        tokenizer=tokenizer,
        max_len=args.max_seq_length)
    train_ds = train_ds.map(trans_func, lazy=False, num_workers=20) # Tokenization
    
    # Prepare data for training
    collator_func = DataCollatorMLM(mask_token_id=tokenizer.mask_token_id, pad_token_id=tokenizer.pad_token_id, token_len=tokenizer.vocab_size) # data collator
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_data_loader = DataLoader(
        dataset=train_ds,
        collate_fn=collator_func,
        num_workers=0,
        batch_sampler=train_batch_sampler,
        return_list=True)

    # Define the model netword and loss
    model = ErnieForMLMPretraining(
            ErnieModel(**ErnieForMLMPretraining.pretrained_init_configuration[
                args.model_name_or_path])) # Train from scratch
    # model = ErnieForMLMPretraining.from_pretrained(
    #     args.model_name_or_path) # Train from check point, continue train
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    ignore_label = IGNORE
    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(train_ds) * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
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

    if args.amp: #mixed precision (fp16)
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    global_step = 0
    last_step = args.num_train_epochs * len(train_ds)
    # if paddle.distributed.get_rank() == 0:
    #     progress_bar = tqdm(total=last_step)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            input_ids, token_type_ids, _, labels = batch
            with paddle.amp.auto_cast(
                        args.amp,
                        #custom_white_list=["layer_norm", "softmax", "gelu"]
                        ):
                logits = model(input_ids=input_ids,
                    token_type_ids=token_type_ids
                    )
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
            # if paddle.distributed.get_rank() == 0:
            #     progress_bar.update(1)
            if global_step % 2000 == 0:
                print(global_step, loss)

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
