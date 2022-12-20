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

import contextlib

import numpy as np
import paddle
from rouge import Rouge

from paddlenlp.metrics import BLEU
from paddlenlp.utils.log import logger


def convert_example(example, text_column, summary_column, tokenizer, max_source_length, max_target_length):
    """
    Convert a example into necessary features.
    """
    inputs = example[text_column]
    targets = example[summary_column]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=False, truncation=True, return_attention_mask=True
    )
    labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(preds, targets):
    assert len(preds) == len(targets), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(preds), len(targets))
    )
    rouge = Rouge()
    bleu4 = BLEU(n_size=4)
    scores = []
    for pred, target in zip(preds, targets):
        try:
            score = rouge.get_scores(" ".join(pred), " ".join(target))
            scores.append([score[0]["rouge-1"]["f"], score[0]["rouge-2"]["f"], score[0]["rouge-l"]["f"]])
        except ValueError:
            scores.append([0, 0, 0])
        bleu4.add_inst(pred, [target])
    rouge1 = np.mean([i[0] for i in scores])
    rouge2 = np.mean([i[1] for i in scores])
    rougel = np.mean([i[2] for i in scores])
    print("\n" + "*" * 15)
    print("The auto evaluation result is:")
    print("rouge-1:", round(rouge1, 4))
    print("rouge-2:", round(rouge2, 4))
    print("rouge-L:", round(rougel, 4))
    print("BLEU-4:", round(bleu4.score(), 4))
    return rougel


@contextlib.contextmanager
def main_process_first(desc="work"):
    if paddle.distributed.get_world_size() > 1:
        rank = paddle.distributed.get_rank()
        is_main_process = rank == 0
        main_process_desc = "main local process"

        try:
            if not is_main_process:
                # tell all replicas to wait
                logger.debug(f"{rank}: waiting for the {main_process_desc} to perform {desc}")
                paddle.distributed.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                logger.debug(f"{rank}: {main_process_desc} completed {desc}, releasing all replicas")
                paddle.distributed.barrier()
    else:
        yield
