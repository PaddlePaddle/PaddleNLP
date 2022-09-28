# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import argparse

from paddlenlp.utils.env import MODEL_HOME


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--task_name",
                        type=str,
                        default='sst-2',
                        help="Task name.")

    parser.add_argument("--optimizer",
                        type=str,
                        default='adadelta',
                        help="Optimizer to use, only support[adam|adadelta].")

    parser.add_argument("--lr",
                        type=float,
                        default=1.0,
                        help="Learning rate for optimizer.")

    parser.add_argument("--num_layers",
                        type=int,
                        default=1,
                        help="Layers number of LSTM.")

    parser.add_argument("--emb_dim",
                        type=int,
                        default=300,
                        help="Embedding dim.")

    parser.add_argument("--output_dim",
                        type=int,
                        default=2,
                        help="Number of classifications.")

    parser.add_argument("--hidden_size",
                        type=int,
                        default=300,
                        help="Hidden size of LSTM")

    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Batch size of training.")

    parser.add_argument("--max_epoch",
                        type=int,
                        default=12,
                        help="Max number of epochs for training.")

    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="Max length for sentence.")

    parser.add_argument(
        "--n_iter",
        type=int,
        default=20,
        help="Number of iterations for one sample in data augmentation.")

    parser.add_argument("--dropout_prob",
                        type=float,
                        default=0.0,
                        help="Drop probability.")

    parser.add_argument("--init_scale",
                        type=float,
                        default=0.1,
                        help="Init scale for parameter")

    parser.add_argument("--log_freq",
                        type=int,
                        default=10,
                        help="The frequency to print evaluation logs.")

    parser.add_argument("--save_steps",
                        type=int,
                        default=100,
                        help="The frequency to print evaluation logs.")

    parser.add_argument("--padding_idx",
                        type=int,
                        default=0,
                        help="The padding index of embedding.")

    parser.add_argument(
        "--model_name",
        type=str,
        default='bert-base-uncased',
        help=
        "Teacher model's name. Maybe its tokenizer would be loaded and used by small model."
    )

    parser.add_argument("--teacher_dir",
                        type=str,
                        help="Teacher model's directory.")

    parser.add_argument("--vocab_path",
                        type=str,
                        default=os.path.join(MODEL_HOME, 'bert-base-uncased',
                                             'bert-base-uncased-vocab.txt'),
                        help="Student model's vocab path.")

    parser.add_argument("--output_dir",
                        type=str,
                        default='models',
                        help="Directory to save models .")

    parser.add_argument("--init_from_ckpt",
                        type=str,
                        default=None,
                        help="The path of layer and optimizer to be loaded.")

    parser.add_argument(
        "--whole_word_mask",
        action="store_true",
        help=
        "If True, use whole word masking method in data augmentation in distilling."
    )

    parser.add_argument("--embedding_name",
                        type=str,
                        default=None,
                        help="The name of pretrained word embedding.")

    parser.add_argument("--vocab_size",
                        type=int,
                        default=10000,
                        help="Student model's vocab size.")

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Weight balance between cross entropy loss and mean square loss.")

    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        help=
        "Random seed for model parameter initialization, data augmentation and so on."
    )

    parser.add_argument("--device",
                        default="gpu",
                        choices=["gpu", "cpu", "xpu"],
                        help="Device selected for inference.")

    args = parser.parse_args()
    return args
