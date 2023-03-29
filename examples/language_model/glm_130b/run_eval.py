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
import re

import paddle
from modeling import GLM130BModel
from paddle.io import DataLoader
from SwissArmyTransformer import get_tokenizer

from paddlenlp.data import Stack, Tuple

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to pre-trained model")
parser.add_argument("--eval_path", default=None, type=str, required=True, help="The eval file path.", )
parser.add_argument('--cloze_eval', action='store_true', help='Evaluation dataset from `--eval_path` is a cloze task.')
parser.add_argument('--overlapping_eval', type=int, default=32, help='Sliding window for overlapping eval.')
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--seq_length', type=int, default=512, help='Maximum sequence length to process for evaluation.')
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "xpu", "npu"], help="Select cpu, gpu, xpu, npu devices.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
# yapf: enable


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
    beginning_tokens = tokenizer.tokenize(" " + text[:start_idx].strip())
    last_token = tokenizer.tokenize(" " + last_token, add_special_tokens=False)
    return beginning_tokens, last_token


def create_eval_dataset(args):
    val_dataloader = None
    eval_batch_size = args.batch_size
    tokenizer = get_tokenizer(tokenizer_type="icetk-glm-130B")
    if not args.cloze_eval:
        with open(args.eval_path, "rb") as reader:
            entire_data = reader.read().decode("utf-8")
        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = wikitext_detokenizer(entire_data)
        tokenized_data = tokenizer.tokenize(entire_data)
        num_tokenized_tokens = len(tokenized_data)
        print("Original Tokens: %d, Detokenized tokens: %d" % (num_tokenized_tokens, num_original_tokens))
        # val_dataset = LMDataset(args, [tokenized_data], tokenizer, num_original_tokens, num_tokenized_tokens)
    else:
        # val_dataset = LambadaDataset(args, tokenizer, strict=True)
        num_tokenized_tokens = 0
        num_original_tokens = 0

    val_dataset = None
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


def do_eval(args):
    paddle.set_device(args.device)

    GLM130BModel.init_weights = lambda *args, **kwargs: None
    paddle.set_default_dtype("float16")

    # tokenizer = get_tokenizer(tokenizer_type="icetk-glm-130B")

    model = GLM130BModel.from_pretrained(
        args.model_path,
        dtype="float16",
        parallel_output=True,
        load_state_as_np=True,
        tensor_parallel_degree=args.tensor_parallel_degree,
        tensor_parallel_rank=args.tensor_parallel_rank,
        low_cpu_mem_usage=True,
    )
    model.eval()

    """
    tic_eval = time.time()
    eval_data_loader = create_eval_dataset(args)
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
    """


def run():
    args = parser.parse_args()
    do_eval(args)


if __name__ == "__main__":
    run()
