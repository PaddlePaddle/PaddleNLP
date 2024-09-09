# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from .modules.bert_for_question_answering import BertForQuestionAnsweringBenchmark
from .modules.bigru_crf import BiGruCrfBenchmark
from .modules.ernie3_for_sequence_classification import (
    Ernie3ForSequenceClassificationBenchmark,
)
from .modules.ernie_tiny import ErnieTinyBenchmark
from .modules.gpt_for_sequence_classification import (
    GPTForSequenceClassificationBenchmark,
)
from .modules.lr_scheduler import *  # noqa: F403
from .modules.optimizer import *  # noqa: F403

try:
    from .modules.stablediffusion import StableDiffusionBenchmark
except Exception:
    StableDiffusionBenchmark = None
from .modules.t5_for_conditional_generation import T5ForConditionalGenerationBenchmark

__all__ = [
    "MODEL_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "LR_SCHEDULER_REGISTRY",
    "get_training_parser",
    "parse_args_and_model",
]

MODEL_REGISTRY = {
    "lac": BiGruCrfBenchmark,
    "ernie_tiny": ErnieTinyBenchmark,
    "ernie3_for_sequence_classification": Ernie3ForSequenceClassificationBenchmark,
    "bert_for_question_answering": BertForQuestionAnsweringBenchmark,
    "gpt_for_sequence_classification": GPTForSequenceClassificationBenchmark,
    "t5_for_conditional_generation": T5ForConditionalGenerationBenchmark,
    "stablediffusion": StableDiffusionBenchmark,
}

OPTIMIZER_REGISTRY = {
    "adam": AdamBenchmark,  # noqa: F405
    "adamw": AdamWBenchmark,  # noqa: F405
    "sgd": SGDBenchmark,  # noqa: F405
}

LR_SCHEDULER_REGISTRY = {
    "lambda_decay": LambdaDecayBenchmark,  # noqa: F405
    "linear_decay_with_warmup": LinearDecayWithWarmupBenchmark,  # noqa: F405
}


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def get_training_parser():
    parser = get_parser()
    add_dataset_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    return parser


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def parse_args_and_model(parser):
    args, _ = parser.parse_known_args()

    if getattr(args, "optimizer", None) is not None:
        args.optimizer = args.optimizer.lower()
        OPTIMIZER_REGISTRY[args.optimizer].add_args(args, parser)
    else:
        raise ValueError("--optimizer must be specified. ")

    if getattr(args, "model", None) is not None:
        args.model = args.model.lower()
        MODEL_REGISTRY[args.model].add_args(args, parser)
    else:
        raise ValueError("--model must be specified. ")

    if getattr(args, "lr_scheduler", None) is not None:
        args.lr_scheduler = args.lr_scheduler.lower()
        LR_SCHEDULER_REGISTRY[args.lr_scheduler].add_args(args, parser)

    args, _ = parser.parse_known_args()

    return args


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="gpu", help="Device. ")

    parser.add_argument("--model", type=str, default=None, help="Model. ")

    parser.add_argument("--logging_steps", type=int, default=10, help="Print logs after N steps. ")
    parser.add_argument("--seed", type=int, default=None, help="Random generator seed. ")

    parser.add_argument("--use_amp", type=str2bool, nargs="?", const=False, help="Enable AMP. ")
    parser.add_argument("--scale_loss", type=float, default=128, help="Loss scale. ")
    parser.add_argument("--amp_level", type=str, default="O2", help="AMP LEVEL. O1 or O2. ")
    parser.add_argument("--amp_use_promote", action="store_true", help="Enable kernel promotion for AMP training. ")
    parser.add_argument("--custom_black_list", type=str, nargs="+", default=None, help="Custom black list for AMP. ")

    parser.add_argument("--to_static", action="store_true", help="Enable to static. ")

    parser.add_argument("--max_steps", type=int, default=None, help="Maximum steps. ")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs. ")

    parser.add_argument("--generated_inputs", action="store_true", help="Use generated inputs. ")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="num_workers of dataloader. When paddlepaddle<=2.4.1, if we use dynamicTostatic mode, we need set num_workeks > 0 ",
    )

    # For benchmark.
    parser.add_argument(
        "--profiler_options",
        type=str,
        default=None,
        help='The option of profiler, which should be in format "key1=value1;key2=value2;key3=value3".',
    )
    parser.add_argument("--save_model", type=str, default=None, help="Directory to save models. ")

    return parser


def add_dataset_args(parser):
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size. ")
    parser.add_argument(
        "--max_seq_len", type=int, default=64, help="Maximum number of tokens in the source sequence. "
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data. ")
    parser.add_argument("--pad_to_max_seq_len", action="store_true", help="Pad to max seq len. ")


def add_optimization_args(parser):
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer. ")
    parser.add_argument("--learning_rate", type=float, default=0.25, help="Learning rate. ")
    parser.add_argument("--lr_scheduler", type=str, default=None, help="Learning rate scheduler")

    parser.add_argument("--scheduler_update_by_epoch", action="store_true", help="Scheduler update after each epoch. ")


def add_model_args(parser):
    parser = parser.add_argument_group()
