#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import argparse
from tqdm import tqdm

import paddle
from paddle.amp import GradScaler, auto_cast
from paddle.optimizer import AdamW
from paddlenlp.transformers import (
    BertForQuestionAnswering,
    BertTokenizer,
    ErnieForQuestionAnswering,
    ErnieTokenizer,
)
from paddlenlp.transformers import ChineseBertForQuestionAnswering, ChineseBertTokenizer

from dataset_cmrc2018 import get_dev_dataloader, get_train_dataloader
from metric_cmrc import compute_prediction, squad_evaluate
from utils import (
    CrossEntropyLossForSQuAD,
    get_scheduler,
    get_writer,
    save_json,
    set_seed,
)
from cmrc_evaluate import get_result

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
    "chinesebert": (ChineseBertForQuestionAnswering, ChineseBertTokenizer),
}


@paddle.no_grad()
def evaluate(model, data_loader, args, output_dir="./"):
    model.eval()
    all_start_logits = []
    all_end_logits = []

    for batch in tqdm(data_loader):
        input_ids, token_type_ids, pinyin_ids = batch
        start_logits_tensor, end_logits_tensor = model(
            input_ids, token_type_ids=token_type_ids, pinyin_ids=pinyin_ids)
        all_start_logits.extend(start_logits_tensor.numpy().tolist())
        all_end_logits.extend(end_logits_tensor.numpy().tolist())

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        data_loader.dataset.data,
        data_loader.dataset.new_data,
        (all_start_logits, all_end_logits),
        False,
        args.n_best_size,
        args.max_answer_length,
        args.null_score_diff_threshold,
    )

    save_json(all_predictions, os.path.join(output_dir, "all_predictions.json"))
    if args.save_nbest_json:
        save_json(all_nbest_json, os.path.join(output_dir,
                                               "all_nbest_json.json"))

    ground_truth_file = os.path.join(args.data_dir, "dev.json")

    eval_results = get_result(ground_truth_file=ground_truth_file,
                              prediction_file=os.path.join(
                                  output_dir, "all_predictions.json"))
    print("CMRC2018 EVALUATE.")
    print(eval_results)
    print("SQUAD EVALUATE.")
    squad_evaluate(
        examples=data_loader.dataset.data,
        preds=all_predictions,
        na_probs=scores_diff_json,
    )
    return eval_results


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.dirname(args.output_dir),
                             "train_cmrc2018.log"),
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")
    paddle.set_device(args.device)
    set_seed(args.seed)
    writer = get_writer(args)

    # get model and tokenizer
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # get dataloader
    train_dataloader = get_train_dataloader(tokenizer, args)
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
        if not any(nd in n for nd in ["bias", "norm", "LayerNorm"])
    ]

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.98,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=paddle.nn.ClipGradByNorm(clip_norm=args.max_grad_norm),
    )

    loss_fn = CrossEntropyLossForSQuAD()

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

    save_json(vars(args), os.path.join(args.output_dir, "args.json"))
    progress_bar = tqdm(range(args.max_train_steps))

    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0

    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            with auto_cast(args.use_amp,
                           custom_white_list=["layer_norm", "softmax", "gelu"]):
                input_ids, token_type_ids, pinyin_ids, start_positions, end_positions = batch
                logits = model(input_ids,
                               token_type_ids=token_type_ids,
                               pinyin_ids=pinyin_ids)
                loss = (loss_fn(logits, (start_positions, end_positions)) /
                        args.gradient_accumulation_steps)
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
                    eval_results = evaluate(model, dev_dataloader, args,
                                            output_dir)
                    for k, v in eval_results.items():
                        writer.add_scalar(f"eval/{k}", v, global_steps)
                        logger.info(f"  {k} = {v}")
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                logger.info("********** Training Done **********")
                return


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type",
                        default="chinesebert",
                        type=str,
                        help="Type of pre-trained model.")
    parser.add_argument(
        "--model_name_or_path",
        default="ChineseBERT-large",
        type=str,
        help="Path to pre-trained model or shortcut name of model.")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data/cmrc2018",
                        help="the path of cmrc2018 data.")
    parser.add_argument(
        "--output_dir",
        default="outputs",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written. "
        "Default as `outputs`")
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for evaluating.")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="gradient_accumulation_steps.")
    parser.add_argument("--learning_rate",
                        default=4e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_radio",
        default=0.1,
        type=float,
        help=
        "Proportion of training steps to perform linear learning rate warmup for."
    )
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=-1,
                        help="warmup_steps.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=250,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--writer_type",
                        choices=["visualdl", "tensorboard"],
                        default="visualdl",
                        help="writer_type.")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--scheduler_type",
                        choices=["linear", "cosine", "poly"],
                        default="linear",
                        type=str,
                        help="scheduler_type.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=35,
        help=
        "The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=
        "If null_score - best_non_null is greater than the threshold predict null."
    )
    parser.add_argument("--max_query_length",
                        type=int,
                        default=64,
                        help="Max query length.")
    parser.add_argument("--max_answer_length",
                        type=int,
                        default=65,
                        help="Max answer length.")
    parser.add_argument("--use_amp",
                        action="store_true",
                        help="Enable mixed precision training.")
    parser.add_argument("--scale_loss",
                        type=float,
                        default=2**15,
                        help="The value of scale_loss for fp16.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="num_workers.")
    parser.add_argument("--save_nbest_json",
                        action="store_true",
                        help="Enable save nbest json.")

    args = parser.parse_args()

    args.model_type = args.model_type.lower()
    args.logdir = os.path.join(args.output_dir, "logs")
    os.makedirs("caches", exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
