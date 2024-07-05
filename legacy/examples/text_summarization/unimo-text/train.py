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
import json
import math
import os
import time

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.optimizer import AdamW
from utils import compute_metrics, create_data_loader, print_args, select_sum, set_seed

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import (
    LinearDecayWithWarmup,
    UNIMOLMHeadModel,
    UNIMOTokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="unimo-text-1.0-summary",
        help="The path or shortcut name of the pre-trained model.",
    )
    parser.add_argument("--train_file", type=str, required=False, default=None, help="Train data path.")
    parser.add_argument("--eval_file", type=str, required=False, default=None, help="Eval data path.")
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="The directory where the checkpoints will be saved."
    )
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay for optimizer.")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", type=float, default=0.02, help="The number of warmup steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="The max value of grad norm.")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="beta2")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="epsilon")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum sequence length of training.")
    parser.add_argument("--max_dec_len", type=int, default=20, help="The maximum sequence length of decoding.")
    parser.add_argument("--min_dec_len", type=int, default=3, help="The minimal sequence length of decoding.")
    parser.add_argument(
        "--max_target_len", type=int, default=30, help="The maximum target sequence length of training."
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The numbers of returned sequences for one input in generation.",
    )
    parser.add_argument(
        "--decode_strategy", type=str, default="beam_search", help="The decode strategy in generation."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The number of highest probability vocabulary tokens to keep for top-k sampling.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="The value used to module the next token probabilities."
    )
    parser.add_argument("--top_p", type=float, default=1.0, help="The cumulative probability for top-p sampling.")
    parser.add_argument("--num_beams", type=int, default=6, help="The number of beams for beam search.")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.2,
        help="The exponential penalty to the sequence length for beam search.",
    )
    parser.add_argument("--device", type=str, default="gpu", help="The device to select for training the model.")
    parser.add_argument(
        "--output_path", type=str, default="./predict.txt", help="The file path where the infer result will be saved."
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to train the model.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to eval and predict.")
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--scale_loss", type=float, default=2**15, help="The value of scale_loss for fp16.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    args = parser.parse_args()
    return args


def save_ckpt(model, tokenizer, save_dir, name):
    output_dir = os.path.join(save_dir, "model_{}".format(name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Need better way to get inner model of DataParallel
    model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def read_file(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            yield line


def run(args):
    paddle.set_device(args.device)
    world_size = dist.get_world_size()

    if world_size > 1:
        dist.init_parallel_env()
    set_seed(args.seed)

    model = UNIMOLMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer = UNIMOTokenizer.from_pretrained(args.model_name_or_path)

    if world_size > 1:
        model = paddle.DataParallel(model)

    if args.do_train:
        train_ds = load_dataset(read_file, file=args.train_file, lazy=False)
        dev_ds = load_dataset(read_file, file=args.eval_file, lazy=False)

        train_ds, train_data_loader = create_data_loader(train_ds, tokenizer, args, "train")
        dev_ds, dev_data_loader = create_data_loader(dev_ds, tokenizer, args, "test")
        if args.max_steps > 0:
            num_training_steps = args.max_steps
            num_train_epochs = math.ceil(num_training_steps / len(train_data_loader))
        else:
            num_training_steps = len(train_data_loader) * args.epochs
            num_train_epochs = args.epochs

        print(f"num_training_steps: {num_training_steps}, num_train_epochs: {num_train_epochs}")

        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.

        decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]

        optimizer = AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm),
        )
        if args.use_amp:
            scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        step = 0
        total_time = 0.0
        for epoch in range(num_train_epochs):
            print("\nEpoch %d/%d" % (epoch + 1, num_train_epochs))
            batch_start_time = time.time()
            for inputs in train_data_loader:
                step += 1
                labels = inputs[-1]
                with paddle.amp.auto_cast(
                    args.use_amp, custom_white_list=["layer_norm", "softmax", "gelu"], level="O1"
                ):
                    logits = model(*inputs[:-1])
                    labels = paddle.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
                    labels = paddle.nn.functional.label_smooth(labels)
                    loss = F.cross_entropy(logits, labels, soft_label=True)
                if args.use_amp:
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.clear_grad(set_to_zero=False)
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.clear_grad()
                lr_scheduler.step()
                total_time += time.time() - batch_start_time
                if step % args.logging_steps == 0:
                    ppl = paddle.exp(loss)
                    print(
                        "epoch %d - step %d - loss: %.4f - ppl: %.4f - lr: %.7f - %.3fs/step"
                        % (epoch, step, loss, ppl, optimizer.get_lr(), total_time / args.logging_steps)
                    )
                    total_time = 0.0

                if step % args.save_steps == 0 or step == num_training_steps:
                    if dist.get_rank() == 0:
                        save_ckpt(model, tokenizer, args.save_dir, step)
                        print("Saved step {} model.\n".format(step))
                        model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
                        evaluation(model_eval, dev_data_loader, args, tokenizer)
                batch_start_time = time.time()
                if step >= num_training_steps:
                    break
            if step >= num_training_steps:
                break

        print("\nTraining completed.")
    elif args.do_eval:
        dev_ds = load_dataset(read_file, file=args.eval_file, lazy=False)
        dev_ds, dev_data_loader = create_data_loader(dev_ds, tokenizer, args, "test")

        model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
        evaluation(model_eval, dev_data_loader, args, tokenizer)


@paddle.no_grad()
def evaluation(model, data_loader, args, tokenizer):
    print("\nEval begin...")
    model.eval()
    pred_ref = []
    total_time = 0.0
    start_time = time.time()
    for step, inputs in enumerate(data_loader, 1):
        input_ids, token_type_ids, position_ids, attention_mask = inputs
        ids, scores = model.generate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=args.max_dec_len,
            min_length=args.min_dec_len,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            num_return_sequences=args.num_return_sequences,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.mask_token_id,
        )

        total_time += time.time() - start_time
        if step % args.logging_steps == 0:
            print("eval step %d - %.3fs/step" % (step, total_time / args.logging_steps))
            total_time = 0.0

        results = select_sum(ids, scores, tokenizer, args.max_dec_len, args.num_return_sequences)
        pred_ref.extend(results)
        start_time = time.time()

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ref in pred_ref:
            fout.write(ref + "\n")

    print("\nSave inference result into: %s" % args.output_path)

    if "title" in data_loader.dataset[0].keys():
        targets = [example["title"] for example in data_loader.dataset]
        compute_metrics(pred_ref, targets)

    model.train()
    return


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    run(args)
