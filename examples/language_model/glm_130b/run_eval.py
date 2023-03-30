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

import json
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np
import paddle
from modeling import GLM130BModel
from paddle.io import DataLoader
from SwissArmyTransformer import get_tokenizer

from paddlenlp.data import Stack, Tuple
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.utils.log import logger

# from pprint import pprint as print


paddle.set_default_dtype("float16")


@dataclass
class DataArgument:
    model_path: str = field(default=None, metadata={"help": "The model state file path."})
    eval_path: str = field(default=None, metadata={"help": "The eval file path."})
    cloze_eval: bool = field(
        default=False, metadata={"help": "Evaluation dataset from `--eval_path` is a cloze task."}
    )
    overlapping_eval: int = field(default=32, metadata={"help": "Sliding window for overlapping eval."})
    batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for training."})
    seq_length: int = field(default=1024, metadata={"help": "Maximum sequence length to process for evaluation."})


class LM_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, seq_len, pad_idx, mask_idx, sop_idx, overlapping_eval=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.sop_idx = sop_idx
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

    def __getitem__(self, idx):
        start_idx = idx * self.overlapping_eval
        end_idx = start_idx + self.seq_len - 2  # gmask, sop
        tokens = self.tokens[start_idx:end_idx]

        prompt_length = self.seq_len - 1 - self.overlapping_eval
        prompt, text = tokens[:prompt_length], tokens[prompt_length:]

        input_len = len(prompt) + 2

        input_ids = np.array(
            prompt
            + [self.mask_idx, self.sop_idx]
            + text
            + [self.pad_idx] * (self.seq_len - 2 - len(prompt) - len(text))
        )
        attention_mask = np.tri(self.seq_len, self.seq_len, dtype=np.int64)
        attention_mask[:input_len, :input_len] = 1
        attention_mask = attention_mask[None, :, :]

        position_ids = np.arange(0, self.seq_len, dtype=np.int64)
        position_ids[input_len:] = np.where(input_ids == self.mask_idx)[0]

        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[input_len : input_len + len(text)] = 1.0

        targets = np.array(input_ids.tolist())

        return [input_ids, loss_mask, attention_mask < 0.5, position_ids, targets]


class Lambada_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, labels, seq_len, pad_idx, mask_idx, sop_idx):
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokens = tokens
        self.labels = labels
        self.mask_idx = mask_idx
        self.sop_idx = sop_idx

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        inputs = self.tokens[idx][-self.seq_len :]
        labels = self.labels[idx]

        inputs = inputs + [self.mask_idx, self.sop_idx]
        input_len = len(inputs)
        input_ids = inputs + labels[:-1]
        if len(input_ids) > self.seq_len - 1:
            input_ids = input_ids[-(self.seq_len - 1) :]
        pred_index = len(input_ids)
        if len(input_ids) < self.seq_len:
            input_ids = input_ids + [self.pad_idx] * (self.seq_len - len(input_ids))
        input_ids = np.array(input_ids)
        attention_mask = np.tri(self.seq_len, self.seq_len).reshape([1, self.seq_len, self.seq_len])
        attention_mask[:, :input_len] = 1
        position_ids = np.arange(0, self.seq_len, dtype=np.int64)
        position_ids[input_len:] = np.where(input_ids == self.mask_idx)[0]
        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[input_len : input_len + len(labels)] = 1.0
        targets = np.array(input_ids.tolist())
        targets[pred_index] = labels[-1]

        return [input_ids, loss_mask, attention_mask < 0.5, position_ids, targets]


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
        tokens = tokenizer.tokenize(text)
        return tokens[:-1], [tokens[-1]]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = tokenizer.tokenize(text[:start_idx].strip())
    last_token = tokenizer.tokenize(" " + last_token)
    return beginning_tokens, last_token


def create_eval_dataset(args):
    val_dataloader = None
    eval_batch_size = args.batch_size
    seq_len = args.seq_length

    tokenizer = get_tokenizer(tokenizer_type="icetk-glm-130B")
    if not args.cloze_eval:
        with open(args.eval_path, "rb") as reader:
            entire_data = reader.read().decode("utf-8")
        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = wikitext_detokenizer(entire_data)
        tokenized_data = tokenizer.tokenize(entire_data)
        num_tokenized_tokens = len(tokenized_data)
        print("Original Tokens: %d, Detokenized tokens: %d" % (num_tokenized_tokens, num_original_tokens))
        val_dataset = LM_Eval_Dataset(
            tokenized_data,
            seq_len,
            0,  # tokenizer.get_command("[pad]"),
            tokenizer.get_command("[gMASK]"),
            tokenizer.get_command("sop"),
            args.overlapping_eval,
        )
    else:
        tokenized_data = []
        tokenized_label = []
        with open(args.eval_path, "r") as f:
            for line in f.readlines():
                text = json.loads(line)["text"]
                tokens, labels = get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)
        val_dataset = Lambada_Eval_Dataset(
            tokenized_data,
            tokenized_label,
            seq_len,
            0,  # tokenizer.get_command("[pad]"),
            tokenizer.get_command("[gMASK]"),
            tokenizer.get_command("sop"),
        )
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

    return args, val_dataloader


def do_eval():
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = paddle.distributed.get_rank()
    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tensor_parallel_degree,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    paddle.distributed.fleet.init(is_collective=True, strategy=strategy)

    parser = PdArgumentParser((DataArgument, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)

    # 屏蔽init_weights
    GLM130BModel.init_weights = lambda *args, **kwargs: None
    paddle.set_default_dtype("float16")

    # tokenizer = get_tokenizer(tokenizer_type="icetk-glm-130B")

    model = GLM130BModel.from_pretrained(
        args.model_path,
        load_state_as_np=True,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
        low_cpu_mem_usage=True,
        dtype="float16",
    )

    tic_eval = time.time()
    args, eval_data_loader = create_eval_dataset(args)
    model.eval()
    total_score = 0
    score_name = "loss" if not args.cloze_eval else "number correct"
    with paddle.no_grad():
        for step, batch in enumerate(eval_data_loader):
            tokens, loss_mask, attention_mask, position_ids, labels = batch
            preds = model(tokens, position_ids, attention_mask)
            if isinstance(preds, tuple):
                preds = preds[0]
            if not args.cloze_eval:
                masked_lm_loss = paddle.nn.functional.cross_entropy(preds, labels, reduction="none")
                loss = paddle.sum(masked_lm_loss * loss_mask)
                total_score += loss.numpy() / (args.num_tokenized_tokens - 1)
            else:
                outputs = paddle.argmax(preds, -1)
                acc = paddle.cast(outputs == labels, "float32")
                acc = paddle.where(paddle.cast(loss_mask, "bool"), acc, paddle.ones_like(acc))
                acc = paddle.sum(paddle.prod(acc, -1))
                total_score += acc.numpy()
            if step % training_args.logging_steps == 0:
                logger.info(
                    "step %d, batch: %d, %s: %f, speed: %.2f step/s"
                    % (step, step, score_name, total_score, training_args.logging_steps / (time.time() - tic_eval))
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
    do_eval()
