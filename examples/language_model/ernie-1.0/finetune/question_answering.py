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

import time
import json
from functools import partial

import numpy as np
import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Pad, Stack, Tuple, Dict
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction

from sequence_classification import BaseTrainer
from paddlenlp.utils.log import logger


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(
            input=start_logits, label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(
            input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss


@paddle.no_grad()
def evaluate(model, data_loader, args):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids,
                                                       token_type_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, _, _ = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), False, args.n_best_size,
        args.max_answer_length)

    # Can also write all_nbest_json and scores_diff_json files if needed
    with open('prediction.json', "w", encoding='utf-8') as writer:
        writer.write(
            json.dumps(
                all_predictions, ensure_ascii=False, indent=4) + "\n")

    squad_evaluate(
        examples=data_loader.dataset.data,
        preds=all_predictions,
        is_whitespace_splited=False)

    model.train()


def prepare_train_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    print(len(examples))

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=args.doc_stride,
        max_seq_len=args.max_seq_length)

    # Let's label those examples!
    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example['offset_mapping']

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']

        # Start/end character index of the answer in the text.
        start_char = answer_starts[0]
        end_char = start_char + len(answers[0])

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
        # Minus one more to reach actual text
        token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and
                offsets[token_end_index][1] >= end_char):
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[
                    token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples[i]["start_positions"] = token_start_index - 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples[i]["end_positions"] = token_end_index + 1

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=args.doc_stride,
        max_seq_len=args.max_seq_length)

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ]

    return tokenized_examples


class QA_TRAINING(BaseTrainer):
    def __init__(self, train_ds, dev_ds, model, tokenizer, args):
        super().__init__()
        self.rank = paddle.distributed.get_rank()
        self.train_ds = train_ds
        self.dev_ds = dev_ds
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.dataloader_inner()
        self.prepare_train_config()
        self.print_config()

    def eval(self):
        if self.rank == 0:
            evaluate(self.model, self.dev_dl, self.args)

    def dataloader_inner(self):
        train_batchify_fn = lambda samples, fn=Dict({
            "input_ids": Pad(axis=0, pad_val=self.tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),
            "start_positions": Stack(dtype="int64"),
            "end_positions": Stack(dtype="int64")
        }): fn(samples)

        dev_batchify_fn = lambda samples, fn=Dict({
                "input_ids": Pad(axis=0, pad_val=self.tokenizer.pad_token_id),
                "token_type_ids": Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id)
         }): fn(samples)

        trans_train_fn = partial(
            prepare_train_features, tokenizer=self.tokenizer, args=self.args)
        trans_dev_fn = partial(
            prepare_validation_features,
            tokenizer=self.tokenizer,
            args=self.args)

        self.train_dl = self.create_dataloader(
            self.train_ds, "train", self.args.batch_size, train_batchify_fn,
            trans_train_fn, True)
        self.dev_dl = self.create_dataloader(
            self.dev_ds, "dev", self.args.batch_size, dev_batchify_fn,
            trans_dev_fn, True)

    def train(self):
        criterion = CrossEntropyLossForSQuAD()

        global_step = 0
        tic_train = time.time()
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(self.train_dl):
                global_step += 1
                input_ids, token_type_ids, start_positions, end_positions = batch
                logits = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids)
                loss = criterion(logits, (start_positions, end_positions))

                if global_step % self.args.logging_steps == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch + 1, step + 1, loss,
                           self.args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_grad()

                if global_step == self.args.num_training_steps:
                    break
