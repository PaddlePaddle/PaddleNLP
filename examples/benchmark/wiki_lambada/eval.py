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
import re
import time
from pprint import pprint as print

# from paddle.distributed.apis import env
import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader

from paddlenlp.data import Stack, Tuple
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.log import logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=False, help="Model type selected in the list")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )

    # only support tensor_parallel_degree
    parser.add_argument(
        "--tensor_parallel_degree",
        type=int,
        default=1,
        help="Model Parallelism degree. Spliting the linear layers to many cards.",
    )

    # Other config
    parser.add_argument("--seed", type=int, default=1024, help="Random seed for initialization")
    parser.add_argument("--sample_nums", type=int, default=16, help="Random seed for initialization")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "xpu", "npu", "gcu"],
        help="select cpu, gpu, xpu, gcu devices.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["bfloat16", "float16", "float32"],
        help="set the dtype of model",
    )
    parser.add_argument(
        "--use_flash_attention",
        type=bool,
        default=False,
        help="Whether to use flash attention",
    )
    # load autodist name files, eg: bloom-176b
    parser.add_argument("--load_autodist", action="store_true", help="whether load auto-dist wieght file")

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
    all_tokens = tokenizer(text.strip())["input_ids"]
    last_token = all_tokens[len(beginning_tokens) :]
    return beginning_tokens, last_token


def create_eval_dataset(args):
    val_dataloader = None
    eval_batch_size = args.batch_size
    seq_len = args.seq_length

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
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
                tokens, labels = get_tokens(tokenizer, text, strict=True)
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


def do_generation():

    # env.set_seed(seed)
    parser = get_eval_parser()
    args = parser.parse_args()
    paddle.set_default_dtype(args.dtype)

    if args.tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "mp_degree": args.tensor_parallel_degree,
        }
        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}
        fleet.init(is_collective=True, strategy=strategy)

    eval_data_loader = create_eval_dataset(args)

    tic_eval = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=args.tensor_parallel_degree,
        tensor_parallel_rank=paddle.distributed.get_rank(),
        use_flash_attention=args.use_flash_attention,
        dtype=args.dtype,  # todo enable set dtype to avoid additional mem usage
    )

    model.eval()
    total_score = 0
    score_name = "loss" if not args.cloze_eval else "number correct"
    eval_data_loader = create_eval_dataset(args)
    with paddle.no_grad():
        for step, batch in enumerate(eval_data_loader):

            tokens, loss_mask = batch[:2]
            labels = batch[-1]
            preds = model(tokens, return_dict=True).logits.detach()
            # cast preds to float32 to keep high-precision
            preds = preds.astype(paddle.float32)

            if not args.cloze_eval:
                masked_lm_loss = paddle.nn.functional.cross_entropy(preds, labels, reduction="none")
                loss = paddle.sum(masked_lm_loss * loss_mask)
                total_score += float(loss) / (args.num_tokenized_tokens - 1)
            else:
                outputs = paddle.argmax(preds, -1)
                acc = paddle.cast(outputs == labels, "float32")
                acc = paddle.where(paddle.cast(loss_mask, "bool"), acc, paddle.ones_like(acc))
                acc = paddle.sum(paddle.prod(acc, -1))
                total_score += float(acc)

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
