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
import time
import random
import argparse
import numpy as np
from scipy import stats
from functools import partial

import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from visualdl import LogWriter

from model import DiffCSE, Encoder
from utils import set_seed, eval_metric
from data import read_text_single, read_text_pair, convert_example, create_dataloader

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "eval", "infer"], default="infer", help="Select which mode to run model, defaults to infer.")
parser.add_argument("--encoder_name", type=str, help="The sentence_encoder name or path that you wanna train based on.")
parser.add_argument("--generator_name", type=str, help="The generator model name or path that you wanna train based on.")
parser.add_argument("--discriminator_name", type=str, help="The discriminator model name or path that you wanna train based on.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--output_emb_size", default=0, type=int, help="Output_embedding_size, 0 means use hidden_size as output embedding size.")
parser.add_argument("--train_set_file", type=str, help="The full path of train_set_file.")
parser.add_argument("--eval_set_file", type=str, help="The full path of eval_set_file.")
parser.add_argument("--infer_set_file", type=str, help="The full path of infer_set_file.")
parser.add_argument("--ckpt_dir", default=None, type=str, help="The ckpt directory where the model checkpoints will be loaded when doing evalution/inference.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, help="The directory where the model checkpoints will be written.")
parser.add_argument("--log_dir", default=None, type=str, help="The directory where log will be written.")
parser.add_argument("--save_infer_path", default="./infer_result.txt", type=str, help="The save directory where the inference result will be written.")
parser.add_argument("--save_steps", type=int, default=10000, help="Step interval for saving checkpoint.")
parser.add_argument("--eval_steps", type=int, default=10000, help="Step interval for evaluation.")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override ecpochs.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--epochs", default=1, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--temp", default=0.05, type=float, help="Temperature for softmax.")
parser.add_argument("--mlm_probability", default=0.15, type=float, help="The ratio for masked language model.")
parser.add_argument("--lambda_weight", default=0.15, type=float, help="The weight for RTD loss.")
parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def do_infer(model, tokenizer, data_loader):
    assert isinstance(
        model, Encoder), "please make sure that model is instance of Encoder."
    sims = []
    model.eval()
    with paddle.no_grad():
        for batch in data_loader:
            query_input_ids, query_token_type_ids, query_attention_mask, key_input_ids, key_token_type_ids, key_attention_mask = batch
            cosine_sim = model.cosine_sim(
                query_input_ids=query_input_ids,
                key_input_ids=key_input_ids,
                query_token_type_ids=query_token_type_ids,
                key_token_type_ids=key_token_type_ids,
                query_attention_mask=query_attention_mask,
                key_attention_mask=key_attention_mask,
            )
            sims.append(cosine_sim.numpy())
        sims = np.concatenate(sims, axis=0)
    model.train()
    return sims


def do_eval(model, tokenizer, data_loader):
    assert isinstance(
        model, Encoder), "please make sure that model is instance of Encoder."
    sims, labels = [], []
    model.eval()
    with paddle.no_grad():
        for batch in data_loader:
            query_input_ids, query_token_type_ids, query_attention_mask, key_input_ids, key_token_type_ids, key_attention_mask, label = batch
            cosine_sim = model.cosine_sim(
                query_input_ids=query_input_ids,
                key_input_ids=key_input_ids,
                query_token_type_ids=query_token_type_ids,
                key_token_type_ids=key_token_type_ids,
                query_attention_mask=query_attention_mask,
                key_attention_mask=key_attention_mask,
            )
            sims.append(cosine_sim.numpy())
            labels.append(label.numpy())

    sims = np.concatenate(sims, axis=0)
    labels = np.concatenate(labels, axis=0)
    score = eval_metric(labels, sims)
    model.train()
    return score


def do_train(model, tokenizer, train_data_loader, dev_data_loader, writer=None):
    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    global_step = 0
    best_score = 0.
    tic_train = time.time()
    model = paddle.DataParallel(model)
    model.train()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, query_attention_mask, key_input_ids, key_token_type_ids, key_attention_mask = batch

            loss, rtd_loss = model(query_input_ids,
                                   key_input_ids,
                                   query_token_type_ids=query_token_type_ids,
                                   key_token_type_ids=key_token_type_ids,
                                   query_attention_mask=query_attention_mask,
                                   key_attention_mask=key_attention_mask)

            global_step += 1
            if global_step % (args.eval_steps // 10) == 0 and rank == 0:
                print(
                    "global step {}, epoch: {}, batch: {}, loss: {:.5f}, rtd_loss: {:.5f}, rtd_acc: {:.5f}, rtd_rep_acc: {:.5f}, rtd_fix_acc: {:.5f}, pos_avg: {:.5f}, neg_avg: {:.5f}, speed: {:.2f} step/s"
                    .format(global_step, epoch, step, loss.item(),
                            rtd_loss.item(), model._layers.rtd_acc,
                            model._layers.rtd_rep_acc,
                            model._layers.rtd_fix_acc,
                            model._layers.encoder.sim.pos_avg,
                            model._layers.encoder.sim.neg_avg,
                            (args.eval_steps // 10) /
                            (time.time() - tic_train)))
                writer.add_scalar(tag="train/loss",
                                  step=global_step,
                                  value=loss.item())
                writer.add_scalar(tag="train/rtd_loss",
                                  step=global_step,
                                  value=rtd_loss.item())
                writer.add_scalar(tag="train/rtd_acc",
                                  step=global_step,
                                  value=model._layers.rtd_acc)
                writer.add_scalar(tag="train/rtd_rep_acc",
                                  step=global_step,
                                  value=model._layers.rtd_rep_acc)
                writer.add_scalar(tag="train/rtd_fix_acc",
                                  step=global_step,
                                  value=model._layers.rtd_fix_acc)

                tic_train = time.time()

            if global_step % args.eval_steps == 0 and rank == 0:
                score = do_eval(model._layers.encoder, tokenizer,
                                dev_data_loader)
                print("Evaluation - score:{:.5f}".format(score))

                if best_score < score:
                    print(
                        "best checkpoint has been updated: from last best_score {} --> new score {}."
                        .format(best_score, score))
                    best_score = score
                    # save best model
                    save_dir = os.path.join(args.save_dir, "best")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir,
                                                   "model_state.pdparams")
                    paddle.save(model._layers.encoder.state_dict(),
                                save_param_path)
                    tokenizer.save_pretrained(save_dir)

                writer.add_scalar(tag="eval/score",
                                  step=global_step,
                                  value=score)
                model.train()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.save_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir,
                                        "checkpoint_{}".format(global_step))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model._layers.encoder.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)

            if args.max_steps > 0 and global_step >= args.max_steps:
                return model


if __name__ == "__main__":
    # set running environment
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define tokenizer for processing data
    tokenizer = ppnlp.transformers.AutoTokenizer.from_pretrained(
        args.encoder_name)
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    if args.mode == "train":
        start_time = time.time()

        # load data
        train_ds = load_dataset(read_text_single,
                                data_path=args.train_set_file,
                                lazy=False)
        dev_ds = load_dataset(read_text_pair,
                              data_path=args.eval_set_file,
                              lazy=False)
        gen_tokenizer = ppnlp.transformers.AutoTokenizer.from_pretrained(
            args.generator_name)
        dis_tokenizer = ppnlp.transformers.AutoTokenizer.from_pretrained(
            args.discriminator_name)

        # intializing DiffCSE model
        model = DiffCSE(encoder_name=args.encoder_name,
                        generator_name=args.generator_name,
                        discriminator_name=args.discriminator_name,
                        enc_tokenizer=tokenizer,
                        gen_tokenizer=gen_tokenizer,
                        dis_tokenizer=dis_tokenizer,
                        temp=args.temp,
                        output_emb_size=args.output_emb_size,
                        mlm_probability=args.mlm_probability,
                        lambda_weight=args.lambda_weight)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=0),  # attention_mask
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # key_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
            Pad(axis=0, pad_val=0),  # attention_mask
        ): [data for data in fn(samples)]
        dev_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=0),  # attention_mask
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # key_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
            Pad(axis=0, pad_val=0),  # attention_mask
            Stack(dtype="int64"),  # labels
        ): [data for data in fn(samples)]

        train_data_loader = create_dataloader(train_ds,
                                              mode="train",
                                              batch_size=args.batch_size,
                                              batchify_fn=batchify_fn,
                                              trans_fn=trans_func)
        dev_data_loader = create_dataloader(dev_ds,
                                            mode="eval",
                                            batch_size=args.batch_size,
                                            batchify_fn=dev_batchify_fn,
                                            trans_fn=trans_func)

        with LogWriter(logdir=os.path.join(args.log_dir, "scalar")) as writer:
            do_train(model,
                     tokenizer,
                     train_data_loader,
                     dev_data_loader,
                     writer=writer)

        end_time = time.time()
        print("running time {} s".format(end_time - start_time))

    if args.mode == "eval":
        start_time = time.time()
        # initalizing encoder model for eval
        model = Encoder(args.encoder_name,
                        temp=args.temp,
                        output_emb_size=args.output_emb_size)
        # load model from saved checkpoint
        if args.ckpt_dir:
            init_from_ckpt = os.path.join(args.ckpt_dir, "model_state.pdparams")
            if os.path.isfile(init_from_ckpt):
                print(
                    "*************************initializing model from {}*****************************"
                    .format(init_from_ckpt))
                state_dict = paddle.load(init_from_ckpt)
                model.set_dict(state_dict)

        dev_ds = load_dataset(read_text_pair,
                              data_path=args.eval_set_file,
                              lazy=False)

        dev_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=0),  # attention_mask
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # key_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
            Pad(axis=0, pad_val=0),  # attention_mask
            Stack(dtype="int64"),  # labels
        ): [data for data in fn(samples)]

        dev_data_loader = create_dataloader(dev_ds,
                                            mode="eval",
                                            batch_size=args.batch_size,
                                            batchify_fn=dev_batchify_fn,
                                            trans_fn=trans_func)

        score = do_eval(model, tokenizer, dev_data_loader)
        print("Evaluation - score:{:.5f}".format(score))

        end_time = time.time()
        print("running time {} s".format(end_time - start_time))

    if args.mode == "infer":
        start_time = time.time()
        # initalizing encoder model for eval
        model = Encoder(args.encoder_name,
                        temp=args.temp,
                        output_emb_size=args.output_emb_size)
        # load model from saved checkpoint
        if args.ckpt_dir:
            init_from_ckpt = os.path.join(args.ckpt_dir, "model_state.pdparams")
            if os.path.isfile(init_from_ckpt):
                print(
                    "*************************initializing model from {}*****************************"
                    .format(init_from_ckpt))
                state_dict = paddle.load(init_from_ckpt)
                model.set_dict(state_dict)

        infer_ds = load_dataset(read_text_pair,
                                data_path=args.infer_set_file,
                                lazy=False,
                                is_infer=True)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
            Pad(axis=0, pad_val=0),  # attention_mask
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # key_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
            Pad(axis=0, pad_val=0),  # attention_mask
        ): [data for data in fn(samples)]

        infer_data_loader = create_dataloader(infer_ds,
                                              mode="infer",
                                              batch_size=args.batch_size,
                                              batchify_fn=batchify_fn,
                                              trans_fn=trans_func)

        cosin_sim = do_infer(model, tokenizer, infer_data_loader)

        with open(args.save_infer_path, "w", encoding="utf-8") as f:
            for idx, cos in enumerate(cosin_sim):
                msg = "{} --> {}\n".format(idx, cos)
                f.write(msg)
            print("Inference result has been saved to : {}".format(
                args.save_infer_path))

        end_time = time.time()
        print("running time {} s".format(end_time - start_time))
