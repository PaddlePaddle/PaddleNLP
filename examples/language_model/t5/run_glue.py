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

import logging
import math
import os

import paddle
from paddle.amp import GradScaler, auto_cast
from paddle.optimizer import AdamW
from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from data import get_dev_dataloader, get_train_dataloader, get_mnli_dev_dataloader, GLUE_PROCESSED
from utils import GLUE_METRICS, get_scheduler, get_writer, set_seed

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model_name_or_path",
        default="t5-small",
        type=str,
        help="Path to pre-trained model or shortcut name of model.",
    )
    parser.add_argument(
        "--task_name",
        default="sst-2",
        type=str,
        help="task_name.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written. "
        "Default as `outputs`",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluating.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=4,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_radio",
        default=0.1,
        type=float,
        help=
        "Proportion of training steps to perform linear learning rate warmup for.",
    )
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=-1,
                        help="warmup_steps.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=10,
                        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        "--writer_type",
        choices=["visualdl", "tensorboard"],
        default="visualdl",
        help="writer_type.",
    )
    parser.add_argument(
        "--scheduler_type",
        choices=["linear", "cosine", "poly"],
        default="linear",
        type=str,
        help="scheduler_type.",
    )
    parser.add_argument("--use_amp",
                        action="store_true",
                        help="Enable mixed precision training.")
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers.",
    )
    parser.add_argument("--is_test", action="store_true", help="is_test.")
    args = parser.parse_args()
    args.task_name = args.task_name.lower()
    args.logdir = os.path.join(args.output_dir, "logs")
    os.makedirs("caches", exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    return args


label_length_map = {
    "cola": 4,
    "sst-2": 1,
    "mrpc": 5,
    "sts-b": 5,
    "qqp": 5,
    "mnli": 4,
    "qnli": 5,
    "rte": 5,
}

logger = logging.getLogger(__name__)


@paddle.no_grad()
def evaluate(model,
             data_loader,
             tokenizer,
             label2id,
             metric_list,
             generate_max_length=5):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in data_loader:
        source_ids, source_mask, labels, target_mask = batch
        outputs = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=generate_max_length,
        )[0]

        for p, l, m in zip(outputs.numpy(), labels.numpy(),
                           target_mask.numpy()):
            pred = tokenizer.decode(p, skip_special_tokens=True).strip()
            label = tokenizer.decode(l[m.astype("bool")],
                                     skip_special_tokens=True).strip()
            if label2id:
                pred = label2id[pred]
                label = label2id[label]
            else:
                pred = float(pred.replace(" ", ""))
                label = float(label.replace(" ", ""))

            all_preds.append(pred)
            all_labels.append(label)

    results = {}
    for metric in metric_list:
        results.update(metric(all_labels, all_preds))
    print(results)
    return results


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "run.log"),
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")
    set_seed(args)

    # metric and label
    label_name = GLUE_PROCESSED[args.task_name][1]
    if label_name:
        label2id = dict(zip(label_name, range(len(label_name))))
    else:
        label2id = None
    metric_list = GLUE_METRICS[args.task_name]
    generate_max_length = label_length_map[args.task_name]

    writer = get_writer(args)

    # get model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    # get dataloader
    train_dataloader = get_train_dataloader(tokenizer, args)
    if args.task_name == "mnli":
        dev_dataloader_match = get_mnli_dev_dataloader(tokenizer,
                                                       args,
                                                       matched=True)
        dev_dataloader_mismatch = get_mnli_dev_dataloader(tokenizer,
                                                          args,
                                                          matched=False)
    else:
        dev_dataloader = get_dev_dataloader(tokenizer, args)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(args.max_train_steps /
                                          num_update_steps_per_epoch)
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # get lr_scheduler
    lr_scheduler = get_scheduler(
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler_type,
        num_warmup_steps=args.warmup_steps
        if args.warmup_steps > 0 else args.warmup_radio,
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    if args.use_amp:
        scaler = GradScaler(init_loss_scaling=args.scale_loss)

    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous train batch size = {args.train_batch_size}")
    logger.info(f"  Instantaneous eval batch size = {args.eval_batch_size}")
    logger.info(
        f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps))

    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0

    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            with auto_cast(args.use_amp,
                           custom_white_list=["layer_norm", "softmax"]):
                source_ids, source_mask, labels, target_mask = batch
                outputs = model(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    labels=labels,
                    decoder_attention_mask=target_mask,
                )
                loss = outputs[0] / args.gradient_accumulation_steps
                tr_loss += loss.item()

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1):
                if args.use_amp:
                    scaler.minimize(optimizer, loss)
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.clear_grad()
                progress_bar.update(1)
                global_steps += 1

                if args.logging_steps > 0 and global_steps % args.logging_steps == 0:
                    writer.add_scalar("lr", lr_scheduler.get_lr(), global_steps)
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps,
                    )
                    logger.info(
                        "global_steps {} - lr: {:.10f}  loss: {:.10f}".format(
                            global_steps,
                            lr_scheduler.get_lr(),
                            (tr_loss - logging_loss) / args.logging_steps,
                        ))
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_steps % args.save_steps == 0:
                    logger.info("********** Running evaluating **********")
                    logger.info(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir,
                                              f"step-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)

                    if args.task_name == "mnli":
                        matched_results = evaluate(
                            model,
                            dev_dataloader_match,
                            tokenizer,
                            label2id,
                            metric_list,
                            generate_max_length,
                        )
                        for k, v in matched_results.items():
                            writer.add_scalar(f"eval/matched_{k}", v,
                                              global_steps)
                            logger.info(f"  {k} = {v}")
                        mismatched_results = evaluate(
                            model,
                            dev_dataloader_mismatch,
                            tokenizer,
                            label2id,
                            metric_list,
                            generate_max_length,
                        )
                        for k, v in mismatched_results.items():
                            writer.add_scalar(f"eval/mismatched_{k}", v,
                                              global_steps)
                            logger.info(f"  {k} = {v}")
                    else:
                        eval_results = evaluate(
                            model,
                            dev_dataloader,
                            tokenizer,
                            label2id,
                            metric_list,
                            generate_max_length,
                        )
                        for k, v in eval_results.items():
                            writer.add_scalar(f"eval/{k}", v, global_steps)
                            logger.info(f"  {k} = {v}")
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                logger.info("********** Running evaluating **********")
                logger.info(f"********** Step {global_steps} **********")
                output_dir = os.path.join(args.output_dir,
                                          f"step-{global_steps}")
                os.makedirs(output_dir, exist_ok=True)

                if args.task_name == "mnli":
                    matched_results = evaluate(
                        model,
                        dev_dataloader_match,
                        tokenizer,
                        label2id,
                        metric_list,
                        generate_max_length,
                    )
                    for k, v in matched_results.items():
                        writer.add_scalar(f"eval/matched_{k}", v, global_steps)
                        logger.info(f"  {k} = {v}")
                    mismatched_results = evaluate(
                        model,
                        dev_dataloader_mismatch,
                        tokenizer,
                        label2id,
                        metric_list,
                        generate_max_length,
                    )
                    for k, v in mismatched_results.items():
                        writer.add_scalar(f"eval/mismatched_{k}", v,
                                          global_steps)
                        logger.info(f"  {k} = {v}")
                else:
                    eval_results = evaluate(
                        model,
                        dev_dataloader,
                        tokenizer,
                        label2id,
                        metric_list,
                        generate_max_length,
                    )
                    for k, v in eval_results.items():
                        writer.add_scalar(f"eval/{k}", v, global_steps)
                        logger.info(f"  {k} = {v}")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("********** Evaluating Done **********")
                logger.info("********** Training Done **********")
                return


if __name__ == "__main__":
    args = parse_args()
    main(args)
