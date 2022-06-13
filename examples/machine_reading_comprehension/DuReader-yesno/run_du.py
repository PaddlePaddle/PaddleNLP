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
import math

from functools import partial
import numpy as np
import paddle

from paddle.io import DataLoader
from args import parse_args
import json

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Pad, Stack, Dict
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.transformers import ErnieGramForSequenceClassification, ErnieGramTokenizer
from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
    "ernie_gram": (ErnieGramForSequenceClassification, ErnieGramTokenizer),
    "roberta": (RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def convert_example(example, tokenizer):
    """convert a Dureader-yesno example into necessary features"""

    feature = tokenizer(text=example['question'],
                        text_pair=example['answer'],
                        max_seq_len=args.max_seq_length)
    feature['labels'] = example['labels']
    feature['id'] = example['id']

    return feature


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("accu: %f" % (accu))
    model.train()  # Switch the model to training mode after evaluation


@paddle.no_grad()
def predict(model, data_loader):
    model.eval()
    res = {}
    for batch in data_loader:
        input_ids, segment_ids, qas_id = batch
        logits = model(input_ids, segment_ids)
        qas_id = qas_id.numpy()
        preds = paddle.argmax(logits, axis=1).numpy()
        for i in range(len(preds)):
            res[str(qas_id[i])] = data_loader.dataset.label_list[preds[i]]
    model.train()
    return res


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    set_seed(args)

    train_ds, dev_ds, test_ds = load_dataset('dureader_yesno',
                                             splits=['train', 'dev', 'test'])

    trans_func = partial(convert_example, tokenizer=tokenizer)

    train_batchify_fn = lambda samples, fn=Dict(
        {
            'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
            'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            'labels': Stack(dtype="int64")
        }): fn(samples)

    test_batchify_fn = lambda samples, fn=Dict(
        {
            'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
            'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            'id': Stack()
        }): fn(samples)

    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=train_batchify_fn,
                                   return_list=True)

    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    dev_data_loader = DataLoader(dataset=dev_ds,
                                 batch_sampler=dev_batch_sampler,
                                 collate_fn=train_batchify_fn,
                                 return_list=True)

    test_ds = test_ds.map(trans_func, lazy=True)
    test_batch_sampler = paddle.io.BatchSampler(test_ds,
                                                batch_size=args.batch_size,
                                                shuffle=False)
    test_data_loader = DataLoader(dataset=test_ds,
                                  batch_sampler=test_batch_sampler,
                                  collate_fn=test_batchify_fn,
                                  return_list=True)

    model = model_class.from_pretrained(args.model_name_or_path,
                                        num_classes=len(train_ds.label_list))

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs
    num_train_epochs = math.ceil(num_training_steps / len(train_data_loader))

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
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
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    tic_train = time.time()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, label = batch

            logits = model(input_ids=input_ids, token_type_ids=segment_ids)
            loss = criterion(logits, label)

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
                    evaluate(model, metric, dev_data_loader)
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

    if rank == 0:
        predictions = predict(model, test_data_loader)
        with open('prediction.json', "w") as writer:
            writer.write(
                json.dumps(predictions, ensure_ascii=False, indent=4) + "\n")


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
