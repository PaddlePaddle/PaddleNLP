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

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a DuReaderYesNo example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        # Account for [CLS], [SEP], [SEP] with "- 3"
        tokens_a, tokens_b = seqs
        max_seq_length -= 3
        while True:  # truncate with longest_first strategy
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_seq_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(
            ([i] * (len(seq) + len(sep))
             for i, (sep, seq) in enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask
                      for sep, seq, s_mask, mask in zip(
                          separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # get the label
        label = example[-2]
        example = example[:-2]
        #create label maps if classification task
        if label_list:
            label_map = {}
            for (i, l) in enumerate(label_list):
                label_map[l] = i
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)
    else:
        qas_id = example[-1]
        example = example[:-2]
    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = _truncate_seqs(tokens_raw, max_seq_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    valid_length = len(input_ids)

    if not is_test:
        return input_ids, segment_ids, valid_length, label
    else:
        return input_ids, segment_ids, valid_length, qas_id


def evaluate(model, metric, data_loader, do_pred=False):
    model.eval()
    if not do_pred:
        metric.reset()
        for batch in data_loader:
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            correct = metric.compute(logits, labels)
            metric.update(correct)
            accu = metric.accumulate()
        print("accu: %f" % (accu))
    else:
        res = {}
        for batch in data_loader:
            input_ids, segment_ids, qas_id = batch
            logits = model(input_ids, segment_ids)
            qas_id = qas_id.numpy()
            preds = paddle.argmax(logits, axis=1).numpy()
            for i in range(len(preds)):
                res[str(qas_id[i])] = data_loader.dataset.get_labels()[preds[i]]
        with open('prediction.json', "w") as writer:
            writer.write(json.dumps(res, ensure_ascii=False, indent=4) + "\n")

    model.train()


def do_train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    set_seed(args)

    train_ds, dev_ds, test_ds = ppnlp.datasets.DuReaderYesNo.get_datasets(
        ['train', 'dev', 'test'])

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=args.max_seq_length)

    train_ds = train_ds.apply(trans_func, lazy=True)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack(),  # length
        Stack(dtype="int64"),  # start_pos
    ): [data for i, data in enumerate(fn(samples)) if i != 2]

    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

    dev_ds = dev_ds.apply(trans_func, lazy=True)

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)

    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

    test_trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=args.max_seq_length,
        is_test=True)

    test_ds = test_ds.apply(test_trans_func, lazy=True)
    test_batch_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=args.batch_size, shuffle=False)

    test_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack()  # length
    ): fn(samples)

    test_data_loader = DataLoader(
        dataset=test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

    model = model_class.from_pretrained(args.model_name_or_path, num_classes=3)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_ds.examples) // args.batch_size * args.num_train_epochs

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
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, label = batch

            logits = model(input_ids=input_ids, token_type_ids=segment_ids)
            loss = criterion(logits, label)

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
            evaluate(model, metric, dev_data_loader)

    if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
        evaluate(model, metric, test_data_loader, True)


if __name__ == "__main__":
    args = parse_args()
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
