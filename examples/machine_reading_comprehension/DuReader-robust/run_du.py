# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
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

import os
import random
import time
import json
import math

from functools import partial
import numpy as np
import paddle

from paddle.io import DataLoader
from args import parse_args

from paddlenlp.data import Pad, Stack, Tuple, Dict
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer
from paddlenlp.transformers import ErnieForQuestionAnswering, ErnieTokenizer
from paddlenlp.transformers import ErnieGramForQuestionAnswering, ErnieGramTokenizer
from paddlenlp.transformers import RobertaForQuestionAnswering, RobertaTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from datasets import load_dataset

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
    "ernie_gram": (ErnieGramForQuestionAnswering, ErnieGramTokenizer),
    "roberta": (RobertaForQuestionAnswering, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, raw_dataset, data_loader, args):
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
        raw_dataset, data_loader.dataset, (all_start_logits, all_end_logits),
        False, args.n_best_size, args.max_answer_length)

    # Can also write all_nbest_json and scores_diff_json files if needed
    with open('prediction.json', "w", encoding='utf-8') as writer:
        writer.write(
            json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

    squad_evaluate(examples=[raw_data for raw_data in raw_dataset],
                   preds=all_predictions,
                   is_whitespace_splited=False)

    model.train()


class CrossEntropyLossForSQuAD(paddle.nn.Layer):

    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(input=start_logits,
                                                        label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(input=end_logits,
                                                      label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss


def run(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    set_seed(args)

    train_examples = load_dataset('PaddlePaddle/dureader_robust', split="train")
    dev_examples = load_dataset('PaddlePaddle/dureader_robust',
                                split="validation")

    column_names = train_examples.column_names
    if rank == 0:
        if os.path.exists(args.model_name_or_path):
            print("init checkpoint from %s" % args.model_name_or_path)

    model = model_class.from_pretrained(args.model_name_or_path)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
        # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
        contexts = examples['context']
        questions = examples['question']

        tokenized_examples = tokenizer(questions,
                                       contexts,
                                       stride=args.doc_stride,
                                       max_seq_len=args.max_seq_length)

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples['token_type_ids'][i]

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[
                            token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(
                        token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index +
                                                               1)

        return tokenized_examples

    if args.do_train:
        train_ds = train_examples.map(prepare_train_features,
                                      batched=True,
                                      remove_columns=column_names,
                                      num_proc=4)
        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)
        train_batchify_fn = lambda samples, fn=Dict(
            {
                "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
                "token_type_ids": Pad(axis=0,
                                      pad_val=tokenizer.pad_token_type_id),
                "start_positions": Stack(dtype="int64"),
                "end_positions": Stack(dtype="int64")
            }): fn(samples)

        train_data_loader = DataLoader(dataset=train_ds,
                                       batch_sampler=train_batch_sampler,
                                       collate_fn=train_batchify_fn,
                                       return_list=True)

        num_training_steps = args.max_steps if args.max_steps > 0 else len(
            train_data_loader) * args.num_train_epochs
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))

        lr_scheduler = LinearDecayWithWarmup(args.learning_rate,
                                             num_training_steps,
                                             args.warmup_proportion)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)
        criterion = CrossEntropyLossForSQuAD()

        global_step = 0
        tic_train = time.time()
        for epoch in range(num_train_epochs):
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                input_ids, token_type_ids, start_positions, end_positions = batch
                logits = model(input_ids=input_ids,
                               token_type_ids=token_type_ids)
                loss = criterion(logits, (start_positions, end_positions))

                if global_step % args.logging_steps == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch + 1, step + 1, loss,
                           args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % args.save_steps == 0 or global_step == num_training_steps:
                    if rank == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        print('Saving checkpoint to:', output_dir)
                    if global_step == num_training_steps:
                        break

    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        #NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
        # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
        contexts = examples['context']
        questions = examples['question']

        tokenized_examples = tokenizer(questions,
                                       contexts,
                                       stride=args.doc_stride,
                                       max_seq_len=args.max_seq_length,
                                       return_attention_mask=True)

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples['token_type_ids'][i]
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(
                examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if args.do_predict and rank == 0:
        dev_ds = dev_examples.map(prepare_validation_features,
                                  batched=True,
                                  remove_columns=column_names,
                                  num_proc=4)
        dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)

        dev_batchify_fn = lambda samples, fn=Dict({
            "input_ids":
            Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids":
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        }): fn(samples)

        dev_data_loader = DataLoader(dataset=dev_ds,
                                     batch_sampler=dev_batch_sampler,
                                     collate_fn=dev_batchify_fn,
                                     return_list=True)

        evaluate(model, dev_examples, dev_data_loader, args)


if __name__ == "__main__":
    args = parse_args()
    run(args)
