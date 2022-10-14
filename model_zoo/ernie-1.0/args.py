# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import paddle
from paddlenlp.utils.log import logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args(MODEL_CLASSES):
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
        sum([ list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values() ], [])),)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
        sum([ list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values() ], [])),)
    parser.add_argument("--continue_training", default=False, type=bool,
        help="Pre-training from existing paddlenlp model weights. Default Fasle and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models.")

    # Train I/O config
    parser.add_argument("--input_dir", default=None, type=str, required=True, help="The input directory where the data will be read from.", )
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the training logs and checkpoints will be written.")
    parser.add_argument("--split", type=str, default='949,50,1', help="Train/valid/test data split.")

    parser.add_argument("--binary_head", type=str2bool, default=True, help="True for NSP task.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--micro_batch_size", default=8, type=int, help="Batch size per device for one step training.", )
    parser.add_argument("--global_batch_size", default=None, type=int, help="Global batch size for all training process. None for not check the size is valid. If we only use data parallelism, it should be device_num * micro_batch_size.")

    # Default training config
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--grad_clip", default=0.0, type=float, help="Grad clip for the parameter.")
    parser.add_argument("--max_lr", default=1e-5, type=float, help="The initial max learning rate for Adam.")
    parser.add_argument("--min_lr", default=5e-5, type=float, help="The initial min learning rate for Adam.")
    parser.add_argument("--warmup_rate", default=0.01, type=float, help="Linear warmup over warmup_steps for learing rate.")

    # Adam optimizer config
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="The beta1 for Adam optimizer. The exponential decay rate for the 1st moment estimates.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="The bate2 for Adam optimizer. The exponential decay rate for the 2nd moment estimates.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    # Training steps config
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.", )
    parser.add_argument("--max_steps", default=500000, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--checkpoint_steps", type=int, default=500, help="Save checkpoint every X updates steps to the model_last folder.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--decay_steps", default=360000, type=int, help="The steps use to control the learing rate. If the step > decay_steps, will use the min_lr.")
    parser.add_argument("--logging_freq", type=int, default=1, help="Log every X updates steps.")
    parser.add_argument("--eval_freq", type=int, default=500, help="Evaluate for every X updates steps.")
    parser.add_argument("--eval_iters", type=int, default=10, help="Evaluate the model use X steps data.")

    # Config for 4D Parallelism
    parser.add_argument("--use_sharding", type=str2bool, nargs='?', const=False, help="Use sharding Parallelism to training.")
    parser.add_argument("--sharding_degree", type=int, default=1, help="Sharding degree. Share the parameters to many cards.")
    parser.add_argument("--dp_degree", type=int, default=1, help="Data Parallelism degree.")
    parser.add_argument("--mp_degree", type=int, default=1, help="Model Parallelism degree. Spliting the linear layers to many cards.")
    parser.add_argument("--pp_degree", type=int, default=1, help="Pipeline Parallelism degree. Spliting the the model layers to different parts.")
    parser.add_argument("--use_recompute", type=str2bool, nargs='?', const=False, help="Using the recompute to save the memory.")

    # AMP config
    parser.add_argument("--use_amp", type=str2bool, nargs='?', const=False, help="Enable mixed precision training.")
    parser.add_argument("--fp16_opt_level", type=str, default="O2", help="Mixed precision training optimization level.")
    parser.add_argument("--enable_addto", type=str2bool, nargs='?', const=True, default=True, help="Whether to enable the addto strategy for gradient accumulation or not. This is only used for AMP training.")
    parser.add_argument("--scale_loss", type=float, default=32768, help="The value of scale_loss for fp16. This is only used for AMP training.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="The hidden dropout prob.")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="The attention probs dropout prob.")

    # Other config
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for initialization.")
    parser.add_argument("--num_workers", type=int, default=2, help="Num of workers for DataLoader.")
    parser.add_argument("--check_accuracy", type=str2bool, nargs='?', const=False, help="Check accuracy for training process.")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "xpu", "mlu"], help="select cpu, gpu, xpu devices.")
    parser.add_argument("--lr_decay_style", type=str, default="cosine", choices=["cosine", "none"], help="Learning rate decay style.")
    parser.add_argument("--share_folder", type=str2bool, nargs='?', const=False, help="Use share folder for data dir and output dir on multi machine.")

    # Argument for bert/ernie
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="Mask token prob.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1, help="Short sequence prob.")
    parser.add_argument("--favor_longer_ngram", type=str2bool, default=False, help="Short sequence prob.")
    parser.add_argument("--max_ngrams", type=int, default=3, help="Short sequence prob.")

    # yapf: enable

    args = parser.parse_args()

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    args.test_iters = args.eval_iters * 10

    if args.check_accuracy:
        if args.hidden_dropout_prob != 0:
            args.hidden_dropout_prob = .0
            logger.warning(
                "The hidden_dropout_prob should set to 0 for accuracy checking."
            )
        if args.attention_probs_dropout_prob != 0:
            args.attention_probs_dropout_prob = .0
            logger.warning(
                "The attention_probs_dropout_prob should set to 0 for accuracy checking."
            )
    if args.dp_degree * args.mp_degree * args.pp_degree * args.sharding_degree == 1:
        if paddle.distributed.get_world_size() > 1:
            args.dp_degree = paddle.distributed.get_world_size()

    return args
