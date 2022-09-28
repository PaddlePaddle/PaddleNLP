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
import re
import os
import json
import math
import time
import argparse

import numpy as np
import paddle
from paddle.io import DataLoader, Dataset
from paddlenlp.transformers import GPTModel, GPTForPretraining
from paddlenlp.transformers import GPTTokenizer
from paddlenlp.transformers import GPTModel
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt": (GPTForPretraining, GPTTokenizer),
}

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(sum([list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values()], [])), )
parser.add_argument("--eval_path", default=None, type=str, required=True, help="The eval file path.", )
parser.add_argument('--cloze_eval', action='store_true', help='Evaluation dataset from `--eval_path` is a cloze task.')
parser.add_argument('--overlapping_eval', type=int, default=32, help='Sliding window for overlapping eval.')
parser.add_argument("--init_checkpoint_path", default=None, type=str, help="The model checkpoint path.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--seq_length', type=int, default=1024, help='Maximum sequence length to process for evaluation.')
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "xpu"], help="Select cpu, gpu, xpu devices.")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
# yapf: enable


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
        self.total_sequences = max(
            math.ceil(targets / self.overlapping_eval) + 1, 1)

    def __len__(self):
        return self.total_sequences

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

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
        tokens = self.tokens[start_idx:end_idx + 1]
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            tokens += [self.pad_idx] * num_pad
        [tokens, loss_mask, attention_mask, position_ids,
         labels] = self._construct_sample(tokens)
        if self.overlapping_eval != self.seq_len and idx != 0:
            loss_mask[:-self.overlapping_eval] *= 0

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
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

        # the pad and eos tokens do not contribute the loss
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        #attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        tokens = self.tokens[idx][:self.seq_len]
        labels = self.labels[idx]
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            tokens += [self.pad_idx] * num_pad
        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[num_tokens - len(labels) - 1:num_tokens - 1] = 1.
        [tokens, attention_mask, position_ids,
         labels] = self._construct_sample(tokens)
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
    last_token = tokenizer(' ' + last_token)["input_ids"]
    return beginning_tokens, last_token


def create_eval_dataset(args):
    val_dataloader = None
    eval_batch_size = args.batch_size
    seq_len = args.seq_length

    tokenizer = GPTTokenizer.from_pretrained(args.model_name)
    if not args.cloze_eval:
        with open(args.eval_path, "rb") as reader:
            entire_data = reader.read().decode('utf-8')
        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = wikitext_detokenizer(entire_data)
        tokenized_data = tokenizer(entire_data)["input_ids"]
        num_tokenized_tokens = len(tokenized_data)
        print('Original Tokens: %d, Detokenized tokens: %d' %
              (num_tokenized_tokens, num_original_tokens))
        val_dataset = LM_Eval_Dataset(tokenized_data, seq_len,
                                      tokenizer.pad_token_id,
                                      args.overlapping_eval)
    else:
        tokenized_data = []
        tokenized_label = []
        with open(args.eval_path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)
        val_dataset = Lambada_Eval_Dataset(tokenized_data, tokenized_label,
                                           seq_len, tokenizer.pad_token_id)
        num_tokenized_tokens = 0
        num_original_tokens = 0

    args.num_examples = len(val_dataset)
    args.num_original_tokens = num_original_tokens
    args.num_tokenized_tokens = num_tokenized_tokens
    val_dataloader = DataLoader(val_dataset,
                                batch_size=eval_batch_size,
                                drop_last=False,
                                collate_fn=Tuple(Stack(), Stack(), Stack(),
                                                 Stack(), Stack()))

    return val_dataloader


def do_eval(args):
    paddle.set_device(args.device)
    model_class, tokenizer_class = MODEL_CLASSES["gpt"]
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    if args.init_checkpoint_path is not None:
        model = GPTForPretraining(
            GPTModel(
                **model_class.pretrained_init_configuration[args.model_name]))

        logger.info("Load model checkpoint from %s" % args.init_checkpoint_path)
        model_dict = paddle.load(os.path.join(args.init_checkpoint_path))
        model.set_dict(model_dict)
    else:
        model = model_class.from_pretrained(args.model_name)

    tic_eval = time.time()
    eval_data_loader = create_eval_dataset(args)
    model.eval()
    total_score = 0
    score_name = "loss" if not args.cloze_eval else "number correct"
    with paddle.no_grad():
        for step, batch in enumerate(eval_data_loader):
            tokens, loss_mask, attention_mask, position_ids, labels = batch
            preds = model(tokens, position_ids, attention_mask)
            if not args.cloze_eval:
                masked_lm_loss = paddle.nn.functional.cross_entropy(
                    preds, labels, reduction="none")
                loss = paddle.sum(masked_lm_loss * loss_mask)
                total_score += loss.numpy() / (args.num_tokenized_tokens - 1)
            else:
                outputs = paddle.argmax(preds, -1)
                acc = paddle.cast(outputs == labels, 'float32')
                acc = paddle.where(paddle.cast(loss_mask, 'bool'), acc,
                                   paddle.ones_like(acc))
                acc = paddle.sum(paddle.prod(acc, -1))
                total_score += acc.numpy()
            if step % args.logging_steps == 0:
                logger.info(
                    "step %d, batch: %d, %s: %f, speed: %.2f step/s" %
                    (step, step, score_name, total_score, args.logging_steps /
                     (time.time() - tic_eval)))
                tic_eval = time.time()

    if not args.cloze_eval:
        total_loss = float(total_score)
        ppl = math.exp(min(20, total_loss))
        token_ratio = (args.num_tokenized_tokens -
                       1) / (args.num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
        string = ' validation results on {} | '.format(args.eval_path)
        string += 'avg loss: {:.4E} | '.format(total_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)
    else:
        num_correct = float(total_score)
        acc = float(num_correct / args.num_examples)
        string = ' validation results on {} | '.format(args.eval_path)
        string += 'number correct: {:.4E} | '.format(num_correct)
        string += 'total examples: {:.4E} | '.format(args.num_examples)
        string += 'avg accuracy: {:.4E}'.format(acc)
    logger.info(string)


if __name__ == "__main__":
    args = parser.parse_args()
    do_eval(args)
