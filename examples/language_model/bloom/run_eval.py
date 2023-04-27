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
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pprint import pprint as print

import numpy as np
import paddle
from paddle import LazyGuard
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.io import DataLoader

from paddlenlp.data import Stack, Tuple
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import AutoTokenizer, BloomForPretraining
from paddlenlp.transformers.bloom.configuration import BloomConfig
from paddlenlp.utils.log import logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list",
    )

    # Train I/O config
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training logs and checkpoints will be written.",
    )
    parser.add_argument("--split", type=str, default="949,50,1", help="Train/valid/test data split.")

    parser.add_argument(
        "--global_batch_size",
        default=None,
        type=int,
        help="Global batch size for all training process. None for not check the size is valid. "
        "If we only use data parallelism, it should be device_num * micro_batch_size.",
    )

    parser.add_argument(
        "--local_batch_size",
        default=None,
        type=int,
        help="Global batch size for all training process. None for not check the size is valid. "
        "If we only use data parallelism, it should be device_num * micro_batch_size.",
    )

    parser.add_argument(
        "--micro_batch_size",
        default=8,
        type=int,
        help="Batch size per device for one step training.",
    )

    # Default training config
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--grad_clip", default=0.0, type=float, help="Grad clip for the parameter.")
    parser.add_argument("--max_lr", default=1e-4, type=float, help="The initial max learning rate for Adam.")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="The initial min learning rate for Adam.")
    parser.add_argument(
        "--warmup_rate", default=0.01, type=float, help="Linear warmup over warmup_steps for learing rate."
    )

    # Adam optimizer config
    parser.add_argument(
        "--adam_beta1",
        default=0.9,
        type=float,
        help="The beta1 for Adam optimizer. The exponential decay rate for the 1st moment estimates.",
    )
    parser.add_argument(
        "--adam_beta2",
        default=0.999,
        type=float,
        help="The bate2 for Adam optimizer. The exponential decay rate for the 2nd moment estimates.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    # Training steps config
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=500000,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--eval_freq", type=int, default=500, help="Evaluate for every X updates steps.")
    parser.add_argument("--eval_iters", type=int, default=10, help="Evaluate the model use X steps data.")
    parser.add_argument(
        "--fuse_transformer",
        type=strtobool,
        default=False,
        help="Whether to use fuse attention and fuse feedforward or not.",
    )

    # Config for 4D Parallelism

    parser.add_argument(
        "--sharding_degree", type=int, default=1, help="Sharding degree. Share the parameters to many cards."
    )

    parser.add_argument("--dp_degree", type=int, default=1, help="Data Parallelism degree.")
    parser.add_argument(
        "--mp_degree", type=int, default=1, help="Model Parallelism degree. Spliting the linear layers to many cards."
    )
    parser.add_argument(
        "--pp_degree",
        type=int,
        default=1,
        help="Pipelines Parallelism degree. Spliting the transformer layers to many cards.",
    )
    parser.add_argument(
        "--use_recompute", type=strtobool, nargs="?", const=False, help="Using the recompute to save the memory."
    )

    parser.add_argument("--lora", type=strtobool, nargs="?", const=False, help="Using LoRA or not.")

    # add sharding stage2/3
    parser.add_argument(
        "--sharding_stage",
        type=int,
        default=1,
        help="sharding stage1/2/3. Stage 1: The optimizer states are partitioned across the processes, "
        "so that each process updates only its partition. Stage 2: The reduced gradients for updating "
        "the model weights are also partitioned such that each process retains only the gradients "
        " corresponding to its portion of the optimizer states. Stage 3: The model parameters are "
        "partitioned across the processes. stage3 will automatically collect and partition them "
        "during the forward and backward passes.",
    )

    parser.add_argument(
        "--sharding_offload", type=strtobool, nargs="?", const=False, help="sharding stage2/3 cpu offload strategy."
    )

    # Pure FP16 config
    parser.add_argument(
        "--use_pure_fp16", type=strtobool, nargs="?", const=False, help="Enable pure fp16 precision training."
    )

    parser.add_argument(
        "--scale_loss",
        type=float,
        default=32768,
        help="The value of scale_loss for fp16. This is only used for AMP training.",
    )

    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="The hidden dropout prob.")
    parser.add_argument(
        "--attention_probs_dropout_prob", type=float, default=0.1, help="The attention probs dropout prob."
    )
    parser.add_argument("--to_static", action="store_true", help="Whether use to_static to train.")

    parser.add_argument("--save_total_limit", type=int, default=3, help="Checkpoint save limit for training.")

    # Other config
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for initialization")
    parser.add_argument(
        "--check_accuracy", type=strtobool, nargs="?", const=False, help="Check accuracy for training process."
    )
    parser.add_argument(
        "--device", type=str, default="gpu", choices=["cpu", "gpu", "xpu", "npu"], help="select cpu, gpu, xpu devices."
    )
    parser.add_argument(
        "--lr_decay_style",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "none"],
        help="Learning rate decay style.",
    )
    parser.add_argument(
        "-p",
        "--profiler_options",
        type=str,
        default=None,
        help='The option of profiler, which should be in format "key1=value1;key2=value2;key3=value3".',
    )

    # only for finetune
    parser.add_argument(
        "--task_name",
        type=str,
        default="sst-2",
        choices=["cola", "sst-2", "mrpc", "sts-b", "qqp", "mnli", "qnli", "rte"],
        help="Task name for finetune.",
    )

    parser.add_argument(
        "--dataset_name",
        default="squad",
        type=str,
        help="The name of the dataset to use. Selected in the list: " + "squad",
    )

    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--max_source_length", type=int, default=512, help="Max sequence length for finetune.")
    parser.add_argument("--max_target_length", type=int, default=512, help="Max sequence length for finetune.")
    return parser


def get_eval_parser():
    parser = get_parser()
    parser.add_argument(
        "--eval_path",
        default=None,
        type=str,
        required=True,
        help="The eval file path.",
    )
    parser.add_argument(
        "--cloze_eval", action="store_true", help="Evaluation dataset from `--eval_path` is a cloze task."
    )
    parser.add_argument("--overlapping_eval", type=int, default=32, help="Sliding window for overlapping eval.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--seq_length", type=int, default=512, help="Maximum sequence length to process for evaluation."
    )
    parser.add_argument("--tensor_parallel_degree", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--tensor_parallel_rank", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="logging step for eval")
    return parser


class LM_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, seq_len, pad_idx, overlapping_eval=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.overlapping_eval = overlapping_eval
        if self.overlapping_eval is None:
            self.overlapping_eval = self.seq_len
        self.overlapping_eval = max(1, self.overlapping_eval)

        self.total_targets = len(self.tokens) - 1
        # remove first sequence tokens
        targets = max(self.total_targets - self.overlapping_eval, 0)
        self.total_sequences = max(math.ceil(targets / self.overlapping_eval) + 1, 1)

    def __len__(self):
        return self.total_sequences

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape((1, seq_length, seq_length))

        # the pad and eos tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[np.where(np.array(tokens) == self.pad_idx)] = 0.0
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        # attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        start_idx = idx * self.overlapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx : end_idx + 1]
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = self.seq_len + 1 - num_tokens
            tokens += [self.pad_idx] * num_pad
        [tokens, loss_mask, attention_mask, position_ids, labels] = self._construct_sample(tokens)
        if self.overlapping_eval != self.seq_len and idx != 0:
            loss_mask[: -self.overlapping_eval] *= 0

        return [tokens, loss_mask, attention_mask, position_ids, labels]


class Lambada_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, labels, seq_len, pad_idx):
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]

        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape((1, seq_length, seq_length))

        # the pad and eos tokens do not contribute the loss
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        # attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, attention_mask, position_ids, labels]

    def __getitem__left_padding(self, idx):
        tokens = self.tokens[idx][: self.seq_len]
        labels = self.labels[idx]
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = self.seq_len + 1 - num_tokens
            # tokens += [self.pad_idx] * num_pad + tokens
            tokens = [self.pad_idx] * num_pad + tokens
        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[-len(labels) :] = 1.0
        [tokens, attention_mask, position_ids, labels] = self._construct_sample(tokens)
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        tokens = self.tokens[idx][: self.seq_len]
        labels = self.labels[idx]
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = self.seq_len + 1 - num_tokens
            tokens += [self.pad_idx] * num_pad
        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[num_tokens - len(labels) - 1 : num_tokens - 1] = 1.0
        [tokens, attention_mask, position_ids, labels] = self._construct_sample(tokens)
        return [tokens, loss_mask, attention_mask, position_ids, labels]


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def get_tokens(tokenizer, text, strict=True):
    if not strict:
        tokens = tokenizer(text)["input_ids"]
        return tokens[:-1], [tokens[-1]]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = tokenizer(text[:start_idx].strip())["input_ids"]
    last_token = tokenizer(" " + last_token)["input_ids"]
    return beginning_tokens, last_token


def create_eval_dataset(args):
    val_dataloader = None
    eval_batch_size = args.batch_size
    seq_len = args.seq_length

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not args.cloze_eval:
        with open(args.eval_path, "rb") as reader:
            entire_data = reader.read().decode("utf-8")
        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = wikitext_detokenizer(entire_data)
        tokenized_data = tokenizer(entire_data)["input_ids"]
        num_tokenized_tokens = len(tokenized_data)
        print("Original Tokens: %d, Detokenized tokens: %d" % (num_tokenized_tokens, num_original_tokens))
        val_dataset = LM_Eval_Dataset(tokenized_data, seq_len, tokenizer.pad_token_id, args.overlapping_eval)
    else:
        tokenized_data = []
        tokenized_label = []
        with open(args.eval_path, "r") as f:
            for line in f.readlines():
                text = json.loads(line)["text"]
                tokens, labels = get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)
        val_dataset = Lambada_Eval_Dataset(tokenized_data, tokenized_label, seq_len, tokenizer.pad_token_id)
        num_tokenized_tokens = 0
        num_original_tokens = 0

    args.num_examples = len(val_dataset)
    args.num_original_tokens = num_original_tokens
    args.num_tokenized_tokens = num_tokenized_tokens
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        drop_last=False,
        collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()),
    )

    return val_dataloader


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank=0):

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)


def load_model(args: str, model_class):
    config: BloomConfig = BloomConfig.from_pretrained(args.model_name_or_path)
    dtype = config.dtype or "float16"
    paddle.set_default_dtype(dtype)

    # Detecting last checkpoint.
    config["enable_fuse_transformer"] = False
    config["use_cache"] = True
    config.use_pure_fp16 = False

    # TODO(wj-Mcat): only support `mp_degree`, so world_size is equal to `world_size`
    world_size = paddle.distributed.get_world_size()

    if world_size == 1:
        return model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            load_state_as_np=True,
            low_cpu_mem_usage=True,
            dtype=dtype,
        )

    # start to init distributed env
    strategy = fleet.DistributedStrategy()

    strategy.hybrid_configs = {
        "dp_degree": getattr(args, "dp_degree", 1),
        "mp_degree": world_size,
        "pp_degree": getattr(args, "pp_degree", 1),
        "sharding_degree": getattr(args, "sharding_degree", 1),
    }

    # Set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

    fleet.init(is_collective=True, strategy=strategy)

    # Obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()

    sharding_size = hcg.get_sharding_parallel_world_size()
    data_world_rank = dp_rank * sharding_size + sharding_rank

    # Seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, data_world_rank, mp_rank)

    config.mp_degree = world_size

    with LazyGuard():
        # init the model without initialized parameters
        model = model_class(config=config)

    weight_file = os.path.join(args.model_name_or_path, f"auto_dist{mp_rank}.pdparams")
    logger.info(f"start to loading sharding model weight file<{weight_file}>")

    # support shard state_dict
    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"sharding model weight file<auto_dist{mp_rank}.pdparams> not found under <{args.model_name_or_path}>"
        )

    state_dict = paddle.load(weight_file, return_numpy=True)
    model.set_state_dict(state_dict)
    return model


def do_generation():
    parser = get_eval_parser()
    args = parser.parse_args()

    eval_data_loader = create_eval_dataset(args)
    tic_eval = time.time()
    model = load_model(args, model_class=BloomForPretraining)
    model.eval()
    total_score = 0
    score_name = "loss" if not args.cloze_eval else "number correct"

    with paddle.no_grad():
        for step, batch in enumerate(eval_data_loader):

            tokens, loss_mask = batch[:2]
            labels = batch[-1]
            with paddle.amp.auto_cast(args.use_pure_fp16):
                preds = model(tokens).detach()

                if not args.cloze_eval:
                    masked_lm_loss = paddle.nn.functional.cross_entropy(preds, labels, reduction="none")
                    masked_lm_loss = paddle.cast(masked_lm_loss, "float32")

                    loss = paddle.sum(masked_lm_loss * loss_mask)
                    total_score += loss.numpy() / (args.num_tokenized_tokens - 1)
                else:
                    outputs = paddle.argmax(preds, -1)
                    acc = paddle.cast(outputs == labels, "float32")
                    acc = paddle.where(paddle.cast(loss_mask, "bool"), acc, paddle.ones_like(acc))
                    acc = paddle.sum(paddle.prod(acc, -1))
                    total_score += acc.numpy()

                if step % args.logging_steps == 0:
                    logger.info(
                        "step %d, batch: %d, %s: %f, speed: %.2f step/s"
                        % (step, step, score_name, total_score, args.logging_steps / (time.time() - tic_eval))
                    )
                    tic_eval = time.time()

    if not args.cloze_eval:
        total_loss = float(total_score)
        ppl = math.exp(min(20, total_loss))
        token_ratio = (args.num_tokenized_tokens - 1) / (args.num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
        string = " validation results on {} | ".format(args.eval_path)
        string += "avg loss: {:.4E} | ".format(total_loss)
        string += "ppl: {:.4E} | ".format(ppl)
        string += "adjusted ppl: {:.4E} | ".format(adjusted_ppl)
        string += "token ratio: {} |".format(token_ratio)
    else:
        num_correct = float(total_score)
        acc = float(num_correct / args.num_examples)
        string = " validation results on {} | ".format(args.eval_path)
        string += "number correct: {:.4E} | ".format(num_correct)
        string += "total examples: {:.4E} | ".format(args.num_examples)
        string += "avg accuracy: {:.4E}".format(acc)
    logger.info(string)


if __name__ == "__main__":
    do_generation()
