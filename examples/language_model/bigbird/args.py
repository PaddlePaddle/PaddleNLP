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
import argparse
import distutils.util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",
                        default="bigbird",
                        type=str,
                        help="Model type selected in training model.")

    parser.add_argument(
        "--model_name_or_path",
        default="bigbird-base-uncased",
        type=str,
        help=
        "Path to pre-trained model or shortcut model name for training model.")

    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="The input directory where the data will be read from.")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )

    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for AdamW.")

    parser.add_argument("--warmup_steps",
                        default=10000,
                        type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Linear warmup proportion over total steps.")

    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help=" Weight decay rate if we apply in the optimizer of Adamw.")

    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for AdamW optimizer.")

    parser.add_argument(
        "--max_steps",
        default=100000,
        type=int,
        help="If > 0: set total number of training steps to perform.")

    parser.add_argument("--logging_steps",
                        type=int,
                        default=1,
                        help="Log every X updates steps.")

    parser.add_argument("--save_steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for initialization.")

    parser.add_argument("--device",
                        type=str,
                        default="gpu",
                        choices=["cpu", "gpu"],
                        help="Select cpu, gpu, xpu devices to train model.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epoches for training.")

    parser.add_argument(
        "--max_encoder_length",
        type=int,
        default=512,
        help=
        "The maximum total input sequence length after SentencePiece tokenization."
    )

    parser.add_argument(
        "--max_pred_length",
        default=75,
        type=int,
        help="The maximum total of masked tokens in input sequence.")

    parser.add_argument(
        "--use_nsp",
        default=False,
        type=bool,
        help="Whether or not add the nsp loss to the total loss.")

    parser.add_argument("--use_amp",
                        type=distutils.util.strtobool,
                        default=False,
                        help="Enable mixed precision training.")

    parser.add_argument(
        "--task_name",
        default="sst-2",
        type=str,
        required=False,
        help=
        "The name of the task to train selected in the list: sst-2, cola, mrpc, sts-b, qqp, mnli, qnli, rte"
    )

    args = parser.parse_args()
    return args
