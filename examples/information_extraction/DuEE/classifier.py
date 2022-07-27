# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
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
"""
classification
"""
import ast
import os
import csv
import json
import warnings
import random
import argparse
import traceback
from functools import partial
from collections import namedtuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import read_by_lines, write_by_lines, load_dict

# warnings.filterwarnings('ignore')
"""
For All pre-trained model（English and Chinese),
Please refer to https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer
"""

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--tag_path", type=str, default=None, help="tag set path")
parser.add_argument("--train_data", type=str, default=None, help="train data")
parser.add_argument("--dev_data", type=str, default=None, help="dev data")
parser.add_argument("--test_data", type=str, default=None, help="test data")
parser.add_argument("--predict_data", type=str, default=None, help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=100, help="validation step")
parser.add_argument("--skip_step", type=int, default=20, help="skip step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoints", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--init_ckpt", type=str, default=None, help="already pretraining model checkpoint")
parser.add_argument("--predict_save_path", type=str, default=None, help="predict data save path")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable.


def set_seed(random_seed):
    """sets random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    paddle.seed(random_seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accuracy = metric.accumulate()
    metric.reset()
    model.train()
    return float(np.mean(losses)), accuracy


def convert_example(example,
                    tokenizer,
                    label_map=None,
                    max_seq_len=512,
                    is_test=False):
    """convert_example"""
    has_text_b = False
    if isinstance(example, dict):
        has_text_b = "text_b" in example.keys()
    else:
        has_text_b = "text_b" in example._fields

    text_b = None
    if has_text_b:
        text_b = example.text_b

    tokenized_input = tokenizer(text=example.text_a,
                                text_pair=text_b,
                                max_seq_len=max_seq_len)
    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']

    if is_test:
        return input_ids, token_type_ids
    else:
        label = np.array([label_map[example.label]], dtype="int64")
        return input_ids, token_type_ids, label


class DuEventExtraction(paddle.io.Dataset):
    """Du"""

    def __init__(self, data_path, tag_path):
        self.label_vocab = load_dict(tag_path)
        self.examples = self._read_tsv(data_path)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)
            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text
                try:
                    example = Example(*line)
                except Exception as e:
                    traceback.print_exc()
                    raise Exception(e)
                examples.append(example)
            return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def data_2_examples(datas):
    """data_2_examples"""
    has_text_b, examples = False, []
    if isinstance(datas[0], list):
        Example = namedtuple('Example', ["text_a", "text_b"])
        has_text_b = True
    else:
        Example = namedtuple('Example', ["text_a"])
    for item in datas:
        if has_text_b:
            example = Example(text_a=item[0], text_b=item[1])
        else:
            example = Example(text_a=item)
        examples.append(example)
    return examples


def do_train():
    paddle.set_device(args.device)
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        "ernie-3.0-medium-zh", num_classes=len(label_map))
    model = paddle.DataParallel(model)
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")

    print("============start train==========")
    train_ds = DuEventExtraction(args.train_data, args.tag_path)
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path)
    test_ds = DuEventExtraction(args.test_data, args.tag_path)

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         label_map=label_map,
                         max_seq_len=args.max_seq_len)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),
        Stack(dtype="int64")  # label
    ): fn(list(map(trans_func, samples)))

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    train_loader = paddle.io.DataLoader(dataset=train_ds,
                                        batch_sampler=batch_sampler,
                                        collate_fn=batchify_fn)
    dev_loader = paddle.io.DataLoader(dataset=dev_ds,
                                      batch_size=args.batch_size,
                                      collate_fn=batchify_fn)
    test_loader = paddle.io.DataLoader(dataset=test_ds,
                                       batch_size=args.batch_size,
                                       collate_fn=batchify_fn)

    num_training_steps = len(train_loader) * args.num_epoch
    metric = paddle.metric.Accuracy()
    criterion = paddle.nn.loss.CrossEntropyLoss()
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    step, best_performerence = 0, 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, token_type_ids, labels) in enumerate(train_loader):
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_item = loss.numpy().item()
            if step > 0 and step % args.skip_step == 0 and rank == 0:
                print(f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) ' \
                    f'- loss: {loss_item:.6f} acc {acc:.5f}')
            if step > 0 and step % args.valid_step == 0 and rank == 0:
                loss_dev, acc_dev = evaluate(model, criterion, metric,
                                             dev_loader)
                print(f'dev step: {step} - loss: {loss_dev:.6f} accuracy: {acc_dev:.5f}, ' \
                        f'current best {best_performerence:.5f}')
                if acc_dev > best_performerence:
                    best_performerence = acc_dev
                    print(f'==============================================save best model ' \
                            f'best performerence {best_performerence:5f}')
                    paddle.save(model.state_dict(),
                                '{}/best.pdparams'.format(args.checkpoints))
            step += 1

    # save the final model
    if rank == 0:
        paddle.save(model.state_dict(),
                    '{}/final.pdparams'.format(args.checkpoints))


def do_predict():
    set_seed(args.seed)
    paddle.set_device(args.device)

    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        "ernie-3.0-medium-zh", num_classes=len(label_map))
    model = paddle.DataParallel(model)
    tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-medium-zh")

    print("============start predict==========")
    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    else:
        state_dict = paddle.load(args.init_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_ckpt)

    # load data from predict file
    sentences = read_by_lines(args.predict_data)  # origin data format
    sentences = [json.loads(sent) for sent in sentences]

    encoded_inputs_list = []
    for sent in sentences:
        sent = sent["text"]
        input_sent = [sent]  # only text_a
        if "text_b" in sent:
            input_sent = [[sent, sent["text_b"]]]  # add text_b
        example = data_2_examples(input_sent)[0]
        input_ids, token_type_ids = convert_example(
            example, tokenizer, max_seq_len=args.max_seq_len, is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [
        encoded_inputs_list[i:i + args.batch_size]
        for i in range(0, len(encoded_inputs_list), args.batch_size)
    ]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        probs_ids = paddle.argmax(probs, -1).numpy()
        probs = probs.numpy()
        for prob_one, p_id in zip(probs.tolist(), probs_ids.tolist()):
            label_probs = {}
            for idx, p in enumerate(prob_one):
                label_probs[id2label[idx]] = p
            results.append({"probs": label_probs, "label": id2label[p_id]})

    assert len(results) == len(sentences)
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    write_by_lines(args.predict_save_path, sentences)
    print("save data {} to {}".format(len(sentences), args.predict_save_path))


if __name__ == '__main__':

    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()
