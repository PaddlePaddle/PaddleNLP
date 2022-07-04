#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""args for classification task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from utils.args import ArgumentGroup

# yapf: disable
parser = argparse.ArgumentParser(__doc__)

# Environment config
model_e = ArgumentGroup(parser, "environment", "Environment settings.")
model_e.add_arg("output_dir",  str, "runs", "Path to save log and checkpoints.")
model_e.add_arg("device", type=str, default="gpu", choices=["cpu", "gpu", "xpu"], help="select cpu, gpu, xpu devices.")
model_e.add_arg("seed", type=int, default=1234, help="Random seed for initialization.")

# Model config
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("init_checkpoint_params", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint_params' has been set, this argument wouldn't be valid.")
model_g.add_arg("num_labels", int, 2, "label number")

# Resource config
model_g.add_arg("unimo_vocab_file", str, './model_files/dict/unimo_en.vocab.txt', "unimo vocab")
model_g.add_arg("encoder_json_file", str, './model_files/dict/unimo_en.encoder.json', 'bpt map')
model_g.add_arg("vocab_bpe_file", str, './model_files/dict/unimo_en.vocab.bpe', "vocab bpe")
model_g.add_arg("unimo_config_path", str, "./model_files/config/unimo_base_en.json",
                "The file to save unimo configuration.")
model_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")


# Training config
train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")
train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("checkpoint_steps", type=int, default=500, help="Save checkpoint every X updates steps to the model_last folder.")
train_g.add_arg("eval_freq", type=int, default=500, help="Evaluate for every X updates steps.")


# Optimization config
train_o = ArgumentGroup(parser, "optimization", "optimization options.")
train_o.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_o.add_arg("decay_proportion", float, 1.0, help="Proportion of training steps to perform learning rate decay.")
train_o.add_arg("warmup_proportion", float, 0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_o.add_arg("grad_clip", default=1.0, type=float, help="Grad clip for the parameter.")
train_o.add_arg("beta1", default=0.9, type=float, help="The beta1 for Adam optimizer. The exponential decay rate for the 1st moment estimates.")
train_o.add_arg("beta2", default=0.98, type=float, help="The bate2 for Adam optimizer. The exponential decay rate for the 2nd moment estimates.")
train_o.add_arg("epsilon", default=1e-06, type=float, help="Epsilon for Adam optimizer.")


# Dataset config
data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_set", str, None, "Path to training data.")
data_g.add_arg("test_set", str, None, "Path to test data.")
data_g.add_arg("dev_set", str, None, "Path to validation data.")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest seqence.")
data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens", bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("num_workers", type=int, default=2, help="Num of workers for DataLoader.")
