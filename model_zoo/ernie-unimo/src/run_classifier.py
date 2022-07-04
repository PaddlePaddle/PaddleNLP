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
"""Finetuning on classification tasks."""
import argparse
import os
import sys
import random
import time
import shutil

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
from visualdl import LogWriter
import paddle.fluid as fluid

from paddlenlp.transformers import LinearAnnealingWithWarmupDecay
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.data import Stack
from paddlenlp.utils.log import logger
from reader.classification_reader import ClassifyReader, pad_batch_records
from model.unimo_finetune import UNIMOConfig, UNIMOModel
from model.tokenization import GptBpeTokenizer
from finetune.classifier import UNIMOClassifier
from utils.init import init_pretraining_params, init_checkpoint
from args.classification_args import parser



def create_pretrained_dataset(
    args,
    tokenizer,
    data_world_size,
    data_world_rank,
    current_step=0,
):
    train_ds = ClassifyReader(args.train_set, tokenizer, args)
    dev_ds = ClassifyReader(args.dev_set, tokenizer, args)
    test_ds = ClassifyReader(args.test_set, tokenizer, args)

    def _collate_data(data, stack_fn=Stack()):
        data = pad_batch_records(data, tokenizer)
        # 0. token_ids,
        # 1. segment_ids,
        # 2. position_ids,
        # 3. input_mask
        # 4. label_id,
        return data

    def loader(dataset, consumed_samples=0):
        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=args.batch_size,
            num_replicas=data_world_size,
            rank=data_world_rank,
            shuffle=False,
            drop_last=True,
            consumed_samples=consumed_samples)
        data_loader = paddle.io.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           num_workers=args.num_workers,
                                           worker_init_fn=None,
                                           collate_fn=_collate_data,
                                           return_list=False)
        return data_loader

    train_dl = loader(train_ds, current_step*args.batch_size * data_world_size)
    dev_dl = paddle.io.DataLoader(dataset=dev_ds,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                worker_init_fn=None,
                                collate_fn=_collate_data,
                                return_list=False)
    test_dl = paddle.io.DataLoader(dataset=test_ds,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                worker_init_fn=None,
                                collate_fn=_collate_data,
                                return_list=False)

    return train_dl, dev_dl, test_dl


@paddle.no_grad()
def run_evaluate(data_loader,
                 model,
                 global_step,
                 args,
                 task_name="valid"):
    model.eval()
    results = []
    labels = []
    for eval_step, batch in enumerate(data_loader):
        # 0. token_ids,
        # 1. segment_ids,
        # 2. position_ids,
        # 3. input_mask
        # 4. label_id,

        input_ids, segment_ids, position_ids, masked_lm_positions, \
        label_ids = batch

        emb_ids = {"word_embedding": input_ids, "sent_embedding": segment_ids, "pos_embedding": position_ids}

        # forward
        prediction_logits  = model(
            emb_ids, masked_lm_positions)
        
        loss, probs = paddle.nn.functional.softmax_with_cross_entropy(logits=prediction_logits, label=label_ids, return_softmax=True)
        results.append(probs)
        labels.append(label_ids)
    probs = paddle.concat(results, axis=0)
    label_ids = paddle.concat(labels, axis=0)
    result = paddle.metric.accuracy(input=probs, label=label_ids, k=1)
    logger.info("{} step {}, accuracy: {} ".format(task_name, global_step, result))
    model.train()
    return result


def set_seed(args):
    if args.device == "cpu":
        idx = 0
    else:
        idx = paddle.distributed.get_rank()
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    paddle.seed(args.seed + idx)


def default_logdir() -> str:
    """
    Same default
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


def do_train(args):
    paddle.set_device(args.device)

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    if worker_num > 1:
        paddle.distributed.init_parallel_env()

    logger.info('{:20}:{}'.format("paddle commit id", paddle.version.commit))
    for arg in vars(args):
        logger.info('{:20}:{}'.format(arg, getattr(args, arg)))

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": worker_num,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1
    }

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()

    # Create the random seed for the worker
    set_seed(args)

    # Create log write,
    log_writer = None
    if worker_index == 0:
        log_writer = LogWriter(os.path.join(args.output_dir, default_logdir()))

    # Define the input data

    tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=args.do_lower_case)
    
    train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(args, tokenizer, worker_num, worker_index)

    
    # Create model 
    model_config = UNIMOConfig(args.unimo_config_path)
    model_config.print_config()
    unimo_pretrain = UNIMOModel(config=model_config)
    if os.path.exists(args.init_pretraining_params):
        checkpoint_path = args.init_pretraining_params
        params_path = os.path.join(checkpoint_path, "model_state.pdparams")
        if os.path.exists(params_path):
            model_dict = paddle.load(params_path)
            unimo_pretrain.encoder.set_state_dict(model_dict)
            logger.info("Load model state dict from init_pretraining_params path: {}.".format(params_path))
        else:
            logger.warning("No model checkpoint file found in %s." %
                           params_path)
        
    model = UNIMOClassifier(unimo_pretrain, args.num_labels)
    

    # Create the learning_rate sheduler and optimizer
    max_steps = len(train_data_loader) * args.epoch * worker_num
    global_step = 0
    logger.info("Total Epochs {}, train dataloader length {},  Max steps {}.".format(args.epoch, len(train_data_loader), max_steps))
    
    assert args.warmup_proportion <= 1.0 and args.warmup_proportion >= 0.0, "warmup_proportion should be in [0, 1]"
    assert args.decay_proportion <= 1.0 and args.decay_proportion >= 0.0, "decay_proportion should be in [0, 1]"
    warmup_steps = args.warmup_proportion * max_steps
    decay_steps = args.decay_proportion * max_steps
    logger.info("Warm up steps {}, Decay steps {}.".format(warmup_steps, decay_steps))
    lr_scheduler = LinearAnnealingWithWarmupDecay(args.learning_rate,
                                                  0,
                                                  warmup_step=warmup_steps,
                                                  decay_step=decay_steps,
                                                  last_epoch=global_step)

    clip = None
    if args.grad_clip > 0:
        clip = paddle.fluid.clip.GradientClipByGlobalNorm(
            clip_norm=args.grad_clip)

    decay_param = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    logger.info("Using paddle.optimizer.AdamW.")
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler if lr_scheduler is not None else args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_param,
        multi_precision=False)

    # load checkpoint vars
    if os.path.exists(args.init_checkpoint_params):
        checkpoint_path = args.init_checkpoint_params
        opt_path = os.path.join(checkpoint_path, "model_state.pdopt")
        params_path = os.path.join(checkpoint_path, "model_state.pdparams")
        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
                
        else:
            logger.warning("No optimizer checkpoint file found in %s." %
                            opt_path)
        if os.path.exists(params_path):
            model_dict = paddle.load(params_path)
            model.set_state_dict(model_dict)
            logger.info("Load model state dict from load_checkpoint path: {}.".format(params_path))
        else:
            logger.warning("No model checkpoint file found in %s." %
                            params_path)

    if paddle.distributed.get_world_size() > 1:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
    
    while True:
        # If not call valid_data_loader, the enumerate will call valid_data_loader
        # many times. and start a new random dataloader.

        # time count
        train_reader_cost = 0.0
        train_run_cost = 0.0
        reader_start = time.time()

        for step, batch in enumerate(train_data_loader()):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()

            # 0. token_ids,
            # 1. segment_ids,
            # 2. position_ids,
            # 3. input_mask
            # 4. label_id,
            # 5. qid,

            # input_ids, segment_ids, position_ids, masked_lm_positions, \
            # label_ids, question_ids = batch
            input_ids, segment_ids, position_ids, masked_lm_positions, \
            label_ids = batch

            emb_ids = {"word_embedding": input_ids, "sent_embedding": segment_ids, "pos_embedding": position_ids}

            # forward
            prediction_logits  = model(emb_ids, masked_lm_positions)
            loss, probs = paddle.nn.functional.softmax_with_cross_entropy(logits=prediction_logits, label=label_ids, return_softmax=True)
            loss = loss.mean()
            if worker_index == 0:
                log_writer.add_scalar(step=global_step, tag="loss", value=loss.numpy())
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            train_run_cost += time.time() - train_start

            global_step += 1

            if lr_scheduler is not None:
                lr_scheduler.step()

            if global_step % args.eval_freq == 0:
                # TODO, check the input data of validation

                accuracy = run_evaluate(valid_data_loader(),
                             model,
                             global_step,
                             args,
                             task_name="valid")
                if worker_index == 0:
                    log_writer.add_scalar(step=global_step, tag="accuracy", value=accuracy.numpy())

            def save_ckpt(output_dir, model, global_step):
                logger.debug("saving models to {}".format(output_dir))
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model

                paddle.save(model.state_dict(),
                            os.path.join(output_dir, "model_state.pdparams"))
                paddle.save(optimizer.state_dict(),
                            os.path.join(output_dir, "model_state.pdopt"))


            if global_step % args.save_steps == 0 or global_step >= max_steps:
                output_dir = os.path.join(args.output_dir,
                                          "model_%d" % global_step)
                if worker_index == 0:
                    save_ckpt(output_dir, model,  global_step)

                if worker_num > 1:
                    paddle.distributed.barrier()

            if global_step % args.checkpoint_steps == 0:
                output_dir = os.path.join(args.output_dir, "model_last")
                if worker_index == 0:
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    output_dir_bak = os.path.join(args.output_dir,
                                                  "model_last_bak")
                    if os.path.exists(output_dir):
                        if os.path.exists(output_dir_bak):
                            shutil.rmtree(output_dir_bak)
                        shutil.move(output_dir, output_dir_bak)
                        os.mkdir(output_dir)
                    save_ckpt(output_dir, model, global_step)

                if worker_num > 1:
                    paddle.distributed.barrier()

            if global_step >= max_steps:
                run_evaluate(test_data_loader(),
                             model,
                             global_step,
                             args,
                             task_name="test")
                del train_data_loader
                return
    

if __name__ == "__main__":
    args = parser.parse_args()
    do_train(args)
