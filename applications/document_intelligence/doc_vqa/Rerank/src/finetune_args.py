#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import argparse

from utils.args import ArgumentGroup

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("ernie_config_path",         str,  None,           "Path to the json file for ernie model config.")
model_g.add_arg("init_checkpoint",          str,  None,           "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params",  str,  None,
                "Init pre-training params which preforms fine-tuning from. If the "
                 "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints",              str,  "checkpoints",  "Path to save checkpoints.")

model_g.add_arg("is_classify",    bool, True,  "is_classify")
model_g.add_arg("is_regression",  bool, False, "is_regression")
model_g.add_arg("task_id",           int,    0,       "task id")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    3,       "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate",     float,  5e-5,    "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion", float,  0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps",        int,    10000,   "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps",  int,    1000,    "The steps interval to evaluate model performance.")
train_g.add_arg("use_recompute",          bool,   False,   "Whether to use recompute optimizer for training.")
train_g.add_arg("use_mix_precision",          bool,   False,   "Whether to use mix-precision optimizer for training.")
train_g.add_arg("use_cross_batch",          bool,   False,   "Whether to use cross-batch for training.")
train_g.add_arg("use_lamb",          bool,   False,   "Whether to use LambOptimizer for training.")
train_g.add_arg("use_dynamic_loss_scaling",    bool,   True,   "Whether to use dynamic loss scaling.")

train_g.add_arg("test_save",            str,    "./checkpoints/test_result",       "test_save")
train_g.add_arg("metric",               str,    "simple_accuracy",   "metric")
train_g.add_arg("incr_every_n_steps",          int,    100,   "Increases loss scaling every n consecutive.")
train_g.add_arg("decr_every_n_nan_or_inf",     int,    2,
                "Decreases loss scaling every n accumulated steps with nan or inf gradients.")
train_g.add_arg("incr_ratio",                  float,  2.0,
                "The multiplier to use when increasing the loss scaling.")
train_g.add_arg("decr_ratio",                  float,  0.8,
                "The less-than-one-multiplier to use when decreasing.")



log_g = ArgumentGroup(parser,     "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")
log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("tokenizer",           str, "FullTokenizer",
              "ATTENTION: the INPUT must be splited by Word with blank while using SentencepieceTokenizer or WordsegTokenizer")
data_g.add_arg("train_set",           str,  None,  "Path to training data.")
data_g.add_arg("test_set",            str,  None,  "Path to test data.")
data_g.add_arg("dev_set",             str,  None,  "Path to validation data.")
data_g.add_arg("vocab_path",          str,  None,  "Vocabulary path.")
data_g.add_arg("max_seq_len",         int,  512,   "Number of words of the longest seqence.")
data_g.add_arg("q_max_seq_len",       int,  32,   "Number of words of the longest seqence.")
data_g.add_arg("p_max_seq_len",       int,  256,   "Number of words of the longest seqence.")
data_g.add_arg("train_data_size",     int,  0,     "Number of training data's total examples. Set for distribute.")
data_g.add_arg("batch_size",          int,  32,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("predict_batch_size",  int,  None,    "Total examples' number in batch for predict. see also --in_tokens.")
data_g.add_arg("in_tokens",           bool, False,
              "If set, the batch size will be the maximum number of tokens in one batch. "
              "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case",       bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("random_seed",         int,  None,     "Random seed.")
data_g.add_arg("label_map_config",    str,  None,  "label_map_path.")
data_g.add_arg("num_labels",          int,  2,     "label number")
data_g.add_arg("diagnostic",          str,  None,  "GLUE Diagnostic Dataset")
data_g.add_arg("diagnostic_save",     str,  None,  "GLUE Diagnostic save f")
data_g.add_arg("max_query_length",          int,   64,    "Max query length.")
data_g.add_arg("max_answer_length",         int,   100,    "Max answer length.")
data_g.add_arg("doc_stride",                int,   128,
               "When splitting up a long document into chunks, how much stride to take between chunks.")
data_g.add_arg("n_best_size",               int,   20,
               "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
data_g.add_arg("chunk_scheme", type=str,  default="IOB", choices=["IO", "IOB", "IOE", "IOBES"], help="chunk scheme")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("is_distributed",    bool,   False,  "If set, then start distributed training.")
run_type_g.add_arg("use_fast_executor",            bool,   False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int,    10,    "Iteration intervals to drop scope.")
run_type_g.add_arg("do_train",                     bool,   True,  "Whether to perform training.")
run_type_g.add_arg("do_val",                       bool,   True,  "Whether to perform evaluation on dev data set.")
run_type_g.add_arg("do_test",                      bool,   True,  "Whether to perform evaluation on test data set.")
run_type_g.add_arg("output_item",                  int,   3,  "Test output format.")
run_type_g.add_arg("output_file_name",             str,   None,  "Test output file name")
run_type_g.add_arg("test_data_cnt",             int,  1110000 ,  "total cnt of testset")
run_type_g.add_arg("use_multi_gpu_test",           bool,   False, "Whether to perform evaluation using multiple gpu cards")
run_type_g.add_arg("metrics",                      bool,   True,  "Whether to perform evaluation on test data set.")
run_type_g.add_arg("shuffle",                      bool,   True,  "")
run_type_g.add_arg("for_cn",                       bool,   False,  "model train for cn or for other langs.")
