# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import time

import paddle
from criterion import Criterion
from evaluate import evaluate
from utils import (
    create_dataloader,
    criteria_map,
    get_label_maps,
    reader,
    save_model_config,
    set_seed,
)

from paddlenlp.datasets import load_dataset
from paddlenlp.layers import (
    GlobalPointerForEntityExtraction,
    GPLinkerForRelationExtraction,
)
from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup
from paddlenlp.utils.log import logger


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args.seed)

    label_maps = get_label_maps(args.task_type, args.label_maps_path)

    train_ds = load_dataset(reader, data_path=args.train_path, lazy=False)
    dev_ds = load_dataset(reader, data_path=args.dev_path, lazy=False)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    train_dataloader = create_dataloader(
        train_ds,
        tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        label_maps=label_maps,
        mode="train",
        task_type=args.task_type,
    )

    dev_dataloader = create_dataloader(
        dev_ds,
        tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        label_maps=label_maps,
        mode="dev",
        task_type=args.task_type,
    )

    encoder = AutoModel.from_pretrained(args.encoder)
    if args.task_type == "entity_extraction":
        model = GlobalPointerForEntityExtraction(encoder, label_maps)
    else:
        model = GPLinkerForRelationExtraction(encoder, label_maps)

    model_config = {"task_type": args.task_type, "label_maps": label_maps, "encoder": args.encoder}

    num_training_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    criterion = Criterion()

    global_step, best_f1 = 1, 0.0
    tr_loss, logging_loss = 0.0, 0.0
    tic_train = time.time()
    for epoch in range(1, args.num_epochs + 1):
        for batch in train_dataloader:
            input_ids, attention_masks, labels = batch

            logits = model(input_ids, attention_masks)

            loss = sum([criterion(o, l) for o, l in zip(logits, labels)]) / 3

            loss.backward()

            tr_loss += loss.item()

            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0 and rank == 0:
                time_diff = time.time() - tic_train
                loss_avg = (tr_loss - logging_loss) / args.logging_steps
                logger.info(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss_avg, args.logging_steps / time_diff)
                )
                logging_loss = tr_loss
                tic_train = time.time()

            if global_step % args.eval_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)
                save_model_config(save_dir, model_config)
                logger.disable()
                tokenizer.save_pretrained(save_dir)
                logger.enable()

                eval_result = evaluate(model, dev_dataloader, label_maps, task_type=args.task_type)
                logger.info("Evaluation precision: " + str(eval_result))

                f1 = eval_result[criteria_map[args.task_type]]
                if f1 > best_f1:
                    logger.info(f"best F1 performance has been updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, "model_state.pdparams")
                    paddle.save(model.state_dict(), save_param_path)
                    save_model_config(save_dir, model_config)
                    logger.disable()
                    tokenizer.save_pretrained(save_dir)
                    logger.enable()
                tic_train = time.time()

            global_step += 1


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=256, type=int, help="The maximum input sequence length.")
    parser.add_argument("--label_maps_path", default="./ner_data/label_maps.json", type=str, help="The file path of the labels dictionary.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay rate for L2 regularizer.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proportion over the training process.")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epoches for training.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--encoder", default="ernie-3.0-mini-zh", type=str, help="Select the pretrained encoder model for GP.")
    parser.add_argument("--task_type", choices=['relation_extraction', 'event_extraction', 'entity_extraction', 'opinion_extraction'], default="entity_extraction", type=str, help="Select the training task type.")
    parser.add_argument("--logging_steps", default=10, type=int, help="The interval steps to logging.")
    parser.add_argument("--eval_steps", default=200, type=int, help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of model parameters for initialization.")

    args = parser.parse_args()
    # yapf: enable

    do_train()
