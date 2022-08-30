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
from distutils.util import strtobool

import tqdm
from paddle.utils.cpp_extension import load


def load_custom_ops():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_dir = cur_dir + "/custom_ops"
    sources = [
        f"{custom_dir}/custom_shape_infer.cc",
        f"{custom_dir}/custom_checkpointoutput.cc",
        f"{custom_dir}/custom_detach.cc", f"{custom_dir}/custom_identity.cc",
        f"{custom_dir}/custom_nll_loss.cc",
        f"{custom_dir}/tied_gather_pattern.cc", f"{custom_dir}/tied_gather.cc",
        f"{custom_dir}/disable_attn_dropout_bwd_pattern.cc",
        f"{custom_dir}/workarounds/prevent_const_expr_folding_op.cc",
        f"{custom_dir}/utils.cc"
    ]
    custom_ops = load(
        name="custom_ops",
        sources=sources,
        extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
        build_directory=custom_dir,
    )
    return custom_ops


class ProgressBar:

    def __init__(self):
        self._bar = None
        self._last = 0

    def __call__(self, progress: int, total: int):
        if self._bar is None:
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            bar_format += "[{elapsed}<{remaining}]"
            self._bar = tqdm.tqdm(desc="Graph compilation",
                                  total=total,
                                  bar_format=bar_format)
        self._bar.update(progress - self._last)
        self._last = progress
        if progress == total:
            self._bar.close()
            self._bar = None


# need to set to 0 when start a new compilation
g_current_progress = 0


def ProgressFunc(progress, total):
    global g_current_progress
    if progress != g_current_progress:
        g_current_progress = progress
        print(f"Graph compilation: {progress}/{total}")


def str_to_bool(val):
    return bool(strtobool(val))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="PRETRAINING",
        help="task",
    )
    parser.add_argument("--input_files",
                        type=str,
                        default="",
                        help="Files to load data from. "
                        "For Pretraining: Path to tfrecord files"
                        "For SQuAD: Path to train-v1.1.json")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--is_training",
                        type=str_to_bool,
                        default=True,
                        help="training or inference")
    # graph
    parser.add_argument("--seq_len",
                        default=128,
                        type=int,
                        help="The sequence length")
    parser.add_argument("--vocab_size",
                        default=30912,
                        type=int,
                        help="Set the size of the vocabulary")
    parser.add_argument(
        "--max_predictions_per_seq",
        default=20,
        type=int,
        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--max_position_embeddings",
                        default=512,
                        type=int,
                        help="the length of the input mask")
    parser.add_argument("--num_hidden_layers",
                        type=int,
                        default=None,
                        help="Override config file if not None")
    parser.add_argument(
        "--hidden_size",
        default=768,
        type=int,
        help="Set the size of the hidden state of the transformer layers size")
    parser.add_argument("--ignore_index",
                        type=int,
                        default=-1,
                        help="ignore mlm index")
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help=
        "Set the layer dropout probability for fully connected layer in embedding and encoder",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.0,
        help="Set the layer dropout probability for attention layer in encoder",
    )
    # optimizer
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--beta1",
                        type=float,
                        default=0.9,
                        help="Set the Adam/Lamb beta1 value")
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999,
                        help="Set the Adam/Lamb beta2 value")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps",
                        default=10,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--scale_loss",
                        type=float,
                        default=1.0,
                        help="The value of scale_loss for fp16.")
    parser.add_argument("--accl1_type",
                        type=str,
                        default='FLOAT',
                        help="FLOAT or FLOAT16")
    parser.add_argument("--accl2_type",
                        type=str,
                        default='FLOAT',
                        help="FLOAT or FLOAT16")
    parser.add_argument("--weight_decay_mode",
                        type=str,
                        default='',
                        help="decay or l2_regularization")
    parser.add_argument("--optimizer_state_offchip",
                        type=str_to_bool,
                        default=True,
                        help="Set the store location of the optimizer tensors")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every X updates steps.")
    # ipu
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="the iteration of the whole dataset",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--micro_batch_size",
                        type=int,
                        default=1,
                        help="micro batch size")
    parser.add_argument("--batches_per_step",
                        type=int,
                        default=1,
                        help="batches per step")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for initialization")
    parser.add_argument("--num_ipus",
                        type=int,
                        default=4,
                        help="Number of IPUs to use")
    parser.add_argument("--ipu_enable_fp16",
                        type=str_to_bool,
                        default=False,
                        help="ipu enable fp16 or not.")
    parser.add_argument("--num_replica",
                        type=int,
                        default=1,
                        help="number of replica")
    parser.add_argument("--enable_grad_acc",
                        type=str_to_bool,
                        default=False,
                        help="enable gradient accumulation")
    parser.add_argument("--grad_acc_factor",
                        type=int,
                        default=1,
                        help="factor of gradient accumulation")
    parser.add_argument(
        "--available_mem_proportion",
        type=float,
        default=0.0,
        help="set the available memory proportion for matmul/conv")
    parser.add_argument("--shuffle",
                        type=str_to_bool,
                        nargs="?",
                        const=True,
                        default=False,
                        help="Shuffle Dataset")
    parser.add_argument("--wandb",
                        type=str_to_bool,
                        nargs="?",
                        const=True,
                        default=False,
                        help="Enable logging to Weights and Biases.")
    parser.add_argument("--enable_load_params",
                        type=str_to_bool,
                        default=False,
                        help="load params or not")
    parser.add_argument("--load_params_path", type=str, help="load params path")
    parser.add_argument(
        "--tf_checkpoint",
        type=str,
        help="Path to Tensorflow Checkpoint to initialise the model.")
    parser.add_argument("--enable_engine_caching",
                        type=str_to_bool,
                        default=True,
                        help="enable engine caching or not")
    args = parser.parse_args()
    return args
