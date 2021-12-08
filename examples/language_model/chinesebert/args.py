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

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        default="chinesebert",
        type=str,
        help="Type of pre-trained model.")
    parser.add_argument(
        "--model_name_or_path",
        default="ChineseBERT-large",
        type=str,
        help="Path to pre-trained model or shortcut name of model.", )
    parser.add_argument(
        "--output_dir",
        default="outputs/cmrc2018",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written. "
        "Default as `outputs`", )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluating.", )

    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="gradient_accumulation_steps.", )
    parser.add_argument(
        "--learning_rate",
        default=4e-5,
        type=float,
        help="The initial learning rate for Adam.", )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
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
        default=2,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_train_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_radio",
        default=0.1,
        type=float,
        help="Proportion of training steps to perform linear learning rate warmup for.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=-1, help="warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Save checkpoint every X updates steps.", )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--writer_type",
        choices=["visualdl", "tensorboard"],
        default="visualdl",
        help="writer_type.", )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Select which device to train model, defaults to gpu.", )
    parser.add_argument(
        "--scheduler_type",
        choices=["linear", "cosine", "poly"],
        default="linear",
        type=str,
        help="scheduler_type.", )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )

    parser.add_argument(
        "--n_best_size",
        type=int,
        default=35,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--max_query_length", type=int, default=64, help="Max query length.")
    parser.add_argument(
        "--max_answer_length", type=int, default=65, help="Max answer length.")

    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable mixed precision training.")

    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.", )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers.", )
    parser.add_argument(
        "--save_nbest_json",
        action="store_true",
        help="Enable save nbest json.")

    args = parser.parse_args()

    args.model_type = args.model_type.lower()
    args.logdir = os.path.join(args.output_dir, "logs")
    os.makedirs("caches", exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    return args
