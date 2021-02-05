# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import os
import random
import time

from functools import partial
import numpy as np
import paddle

from paddle.io import DataLoader
from args import parse_args
import json
import paddlenlp as ppnlp

from paddlenlp.datasets import SQuAD, DuReaderRobust, CMRC, DRCD
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer, ErnieForQuestionAnswering, ErnieTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics.squad import squad_evaluate, compute_predictions

TASK_CLASSES = {"dureader-robust": DuReaderRobust, "cmrc": CMRC, "drcd": DRCD}

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=start_logits, label=start_position, soft_label=False)
        start_loss = paddle.mean(start_loss)
        end_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=end_logits, label=end_position, soft_label=False)
        end_loss = paddle.mean(end_loss)

        loss = (start_loss + end_loss) / 2
        return loss


def evaluate(model, data_loader, args, tokenizer, do_pred=False):
    model.eval()

    RawResult = collections.namedtuple(
        "RawResult", ["unique_id", "start_logits", "end_logits"])

    all_results = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, segment_ids, unipue_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids, segment_ids)

        for idx in range(unipue_ids.shape[0]):
            if len(all_results) % 1000 == 0 and len(all_results):
                print("Processing example: %d" % len(all_results))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()
            unique_id = int(unipue_ids[idx])
            start_logits = [float(x) for x in start_logits_tensor.numpy()[idx]]
            end_logits = [float(x) for x in end_logits_tensor.numpy()[idx]]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

    all_predictions, _, _ = compute_predictions(
        data_loader.dataset.examples, data_loader.dataset.features, all_results,
        args.n_best_size, args.max_answer_length, args.do_lower_case, False,
        0.0, args.verbose, tokenizer, False)
    if do_pred:
        with open('prediction.json', "w", encoding='utf-8') as writer:
            writer.write(
                json.dumps(
                    all_predictions, ensure_ascii=False, indent=4) + "\n")
    else:
        squad_evaluate(
            examples=data_loader.dataset.examples,
            preds=all_predictions,
            is_whitespace_splited=False)

    model.train()


def do_train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    task_name = args.task_name.lower()
    dataset_class = TASK_CLASSES[task_name]

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    root = args.data_path
    set_seed(args)

    train_ds = dataset_class(
        tokenizer=tokenizer,
        root=root,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        max_seq_length=args.max_seq_length,
        mode='train')

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)

    train_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack(),  # unipue_id
        Stack(dtype="int64"),  # start_pos
        Stack(dtype="int64")  # end_pos
    ): [data for i, data in enumerate(fn(samples)) if i != 2]

    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=train_batchify_fn,
        return_list=True)

    dev_ds = dataset_class(
        tokenizer=tokenizer,
        root=root,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        max_seq_length=args.max_seq_length,
        mode='dev')

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)

    dev_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack()  # unipue_id
    ): fn(samples)

    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=dev_batchify_fn,
        return_list=True)

    model = model_class.from_pretrained(args.model_name_or_path)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])
    criterion = CrossEntropyLossForSQuAD()

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, start_positions, end_positions = batch

            logits = model(input_ids=input_ids, token_type_ids=segment_ids)
            loss = criterion(logits, (start_positions, end_positions))

            if global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()

            if global_step % args.save_steps == 0:
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
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

        if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
            evaluate(model, dev_data_loader, args, tokenizer)


if __name__ == "__main__":
    args = parse_args()
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
