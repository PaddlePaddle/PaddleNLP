# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddlenlp.data import Pad, Stack, Dict
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer, ErnieForQuestionAnswering, MPNetForQuestionAnswering, ErnieTokenizer, MPNetTokenizer

from paddlenlp.transformers import LinearDecayWithWarmup, CosineDecayWithWarmup
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from datasets import load_dataset
from collections import OrderedDict

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
    "mpnet": (MPNetForQuestionAnswering, MPNetTokenizer)
}


def _get_layer_lr_radios(layer_decay=0.8, n_layers=12):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = OrderedDict({
        "mpnet.embeddings.": 0,
        "mpnet.encoder.relative_attention_bias.": 0,
        "qa_outputs.": n_layers + 2,
    })
    for layer in range(n_layers):
        key_to_depths[f"mpnet.encoder.layer.{str(layer)}."] = layer + 1
    return {
        key: (layer_decay**(n_layers + 2 - depth))
        for key, depth in key_to_depths.items()
    }


def prepare_train_features(examples, tokenizer, args):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    contexts = examples['context']
    questions = examples['question']

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   max_seq_len=args.max_seq_length,
                                   stride=args.doc_stride,
                                   return_attention_mask=True)

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
        sequence_A_lengths = input_ids.index(tokenizer.sep_token_id) + 2
        sequence_B_lengths = len(input_ids) - sequence_A_lengths
        sequence_ids = [0] * sequence_A_lengths + [1] * sequence_B_lengths

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
                tokenized_examples["start_positions"].append(token_start_index -
                                                             1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, args):
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
        input_ids = tokenized_examples["input_ids"][i]
        sequence_A_lengths = input_ids.index(tokenizer.sep_token_id) + 2
        sequence_B_lengths = len(input_ids) - sequence_A_lengths
        sequence_ids = [0] * sequence_A_lengths + [1] * sequence_B_lengths
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model,
             data_loader,
             raw_dataset,
             args,
             global_step,
             write_predictions=False):
    model.eval()

    all_start_logits = []
    all_end_logits = []

    for batch in data_loader:
        input_ids = batch[0]
        start_logits_tensor, end_logits_tensor = model(input_ids)

        for idx in range(start_logits_tensor.shape[0]):
            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        raw_dataset, data_loader.dataset, (all_start_logits, all_end_logits),
        args.version_2_with_negative, args.n_best_size, args.max_answer_length,
        args.null_score_diff_threshold)

    # Can also write all_nbest_json and scores_diff_json files if needed
    if write_predictions:
        with open(f'{str(global_step)}_prediction.json', "w",
                  encoding='utf-8') as writer:
            writer.write(
                json.dumps(all_predictions, ensure_ascii=False, indent=4) +
                "\n")

    squad_evaluate(examples=[raw_data for raw_data in raw_dataset],
                   preds=all_predictions,
                   na_probs=scores_diff_json)

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
    if rank == 0:
        if os.path.exists(args.model_name_or_path):
            print("init checkpoint from %s" % args.model_name_or_path)

    model = model_class.from_pretrained(args.model_name_or_path)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.do_predict:
        if args.version_2_with_negative:
            dev_examples = load_dataset('squad_v2', split='validation')
        else:
            dev_examples = load_dataset('squad', split='validation')
        column_names = dev_examples.column_names
        dev_ds = dev_examples.map(partial(prepare_validation_features,
                                          tokenizer=tokenizer,
                                          args=args),
                                  batched=True,
                                  remove_columns=column_names,
                                  num_proc=4)
        dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)

        dev_batchify_fn = lambda samples, fn=Dict(
            {"input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id)}): fn(
                samples)

        dev_data_loader = DataLoader(dataset=dev_ds,
                                     batch_sampler=dev_batch_sampler,
                                     collate_fn=dev_batchify_fn,
                                     num_workers=4,
                                     return_list=True)

    if args.do_train:
        # layer_lr for base
        if args.layer_lr_decay != 1.0:
            layer_lr_radios_map = _get_layer_lr_radios(args.layer_lr_decay,
                                                       n_layers=12)
            for name, parameter in model.named_parameters():
                layer_lr_radio = 1.0
                for k, radio in layer_lr_radios_map.items():
                    if k in name:
                        layer_lr_radio = radio
                        break
                parameter.optimize_attr["learning_rate"] *= layer_lr_radio

        if args.version_2_with_negative:
            train_examples = load_dataset('squad_v2', split='train')
        else:
            train_examples = load_dataset('squad', split='train')
        column_names = train_examples.column_names
        train_ds = train_examples.map(partial(prepare_train_features,
                                              tokenizer=tokenizer,
                                              args=args),
                                      batched=True,
                                      remove_columns=column_names,
                                      num_proc=4)
        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)
        train_batchify_fn = lambda samples, fn=Dict(
            {
                "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
                "start_positions": Stack(dtype="int64"),
                "end_positions": Stack(dtype="int64")
            }): fn(samples)

        train_data_loader = DataLoader(dataset=train_ds,
                                       batch_sampler=train_batch_sampler,
                                       collate_fn=train_batchify_fn,
                                       num_workers=4,
                                       return_list=True)

        num_training_steps = args.max_steps if args.max_steps > 0 else len(
            train_data_loader) * args.num_train_epochs
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))

        if args.scheduler_type == "linear":
            lr_scheduler = LinearDecayWithWarmup(args.learning_rate,
                                                 num_training_steps,
                                                 args.warmup_proportion)
        elif args.scheduler_type == "cosine":
            lr_scheduler = CosineDecayWithWarmup(args.learning_rate,
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
            beta1=0.9,
            beta2=0.98,
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
                input_ids, start_positions, end_positions = batch

                logits = model(input_ids=input_ids)
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

                    if args.do_predict and rank == 0:
                        evaluate(model, dev_data_loader, dev_examples, args,
                                 global_step, False)
                        print("=" * 50)

                    if global_step == num_training_steps:
                        return


def print_arguments(args):
    """print arguments"""
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    run(args)
