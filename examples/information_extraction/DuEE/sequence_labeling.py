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
sequence labeling
"""
import ast
import os
import json
import warnings
import random
import argparse
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer
from paddlenlp.metrics import ChunkEvaluator
from utils import read_by_lines, write_by_lines, load_dict

warnings.filterwarnings('ignore')

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


def set_seed(args):
    """sets random seed"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, num_label, data_loader):
    """evaluate"""
    model.eval()
    metric.reset()
    losses = []
    for input_ids, seg_ids, seq_lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        loss = paddle.mean(
            criterion(logits.reshape([-1, num_label]), labels.reshape([-1])))
        losses.append(loss.numpy())
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, seq_lens, preds,
                                                     labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    avg_loss = np.mean(losses)
    model.train()

    return precision, recall, f1_score, avg_loss


def convert_example_to_feature(example,
                               tokenizer,
                               label_vocab=None,
                               max_seq_len=512,
                               no_entity_label="O",
                               ignore_label=-1,
                               is_test=False):
    tokens, labels = example
    tokenized_input = tokenizer(tokens,
                                return_length=True,
                                is_split_into_words=True,
                                max_seq_len=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = tokenized_input['seq_len']

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels = labels[:(max_seq_len - 2)]
        encoded_label = [no_entity_label] + labels + [no_entity_label]
        encoded_label = [label_vocab[x] for x in encoded_label]
        return input_ids, token_type_ids, seq_len, encoded_label


class DuEventExtraction(paddle.io.Dataset):
    """DuEventExtraction"""

    def __init__(self, data_path, tag_path):
        self.label_vocab = load_dict(tag_path)
        self.word_ids = []
        self.label_ids = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                self.word_ids.append(words)
                self.label_ids.append(labels)
        self.label_num = max(self.label_vocab.values()) + 1

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index]


def do_train():
    paddle.set_device(args.device)
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    no_entity_label = "O"
    ignore_label = -1

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        "ernie-3.0-medium-zh", num_classes=len(label_map))
    model = paddle.DataParallel(model)

    print("============start train==========")
    train_ds = DuEventExtraction(args.train_data, args.tag_path)
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path)
    test_ds = DuEventExtraction(args.test_data, args.tag_path)

    trans_func = partial(convert_example_to_feature,
                         tokenizer=tokenizer,
                         label_vocab=train_ds.label_vocab,
                         max_seq_len=args.max_seq_len,
                         no_entity_label=no_entity_label,
                         ignore_label=ignore_label,
                         is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # token type ids
        Stack(dtype='int64'),  # sequence lens
        Pad(axis=0, pad_val=ignore_label, dtype='int64')  # labels
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

    metric = ChunkEvaluator(label_list=train_ds.label_vocab.keys(),
                            suffix=False)
    criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    step, best_f1 = 0, 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, token_type_ids, seq_lens,
                  labels) in enumerate(train_loader):
            logits = model(input_ids,
                           token_type_ids).reshape([-1, train_ds.label_num])
            loss = paddle.mean(criterion(logits, labels.reshape([-1])))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_item = loss.numpy().item()
            if step > 0 and step % args.skip_step == 0 and rank == 0:
                print(
                    f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}'
                )
            if step > 0 and step % args.valid_step == 0 and rank == 0:
                p, r, f1, avg_loss = evaluate(model, criterion, metric,
                                              len(label_map), dev_loader)
                print(f'dev step: {step} - loss: {avg_loss:.5f}, precision: {p:.5f}, recall: {r:.5f}, ' \
                        f'f1: {f1:.5f} current best {best_f1:.5f}')
                if f1 > best_f1:
                    best_f1 = f1
                    print(f'==============================================save best model ' \
                            f'best performerence {best_f1:5f}')
                    paddle.save(model.state_dict(),
                                '{}/best.pdparams'.format(args.checkpoints))
            step += 1

    # save the final model
    if rank == 0:
        paddle.save(model.state_dict(),
                    '{}/final.pdparams'.format(args.checkpoints))


def do_predict():
    paddle.set_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        "ernie-3.0-medium-zh", num_classes=len(label_map))

    no_entity_label = "O"
    ignore_label = len(label_map)

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
        sent = sent["text"].replace(" ", "\002")
        input_ids, token_type_ids, seq_len = convert_example_to_feature(
            [list(sent), []],
            tokenizer,
            max_seq_len=args.max_seq_len,
            is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids, seq_len))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # input_ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # token_type_ids
        Stack(dtype='int64')  # sequence lens
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [
        encoded_inputs_list[i:i + args.batch_size]
        for i in range(0, len(encoded_inputs_list), args.batch_size)
    ]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, token_type_ids, seq_lens = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=-1)
        probs_ids = paddle.argmax(probs, -1).numpy()
        probs = probs.numpy()
        for p_list, p_ids, seq_len in zip(probs.tolist(), probs_ids.tolist(),
                                          seq_lens.tolist()):
            prob_one = [
                p_list[index][pid]
                for index, pid in enumerate(p_ids[1:seq_len - 1])
            ]
            label_one = [id2label[pid] for pid in p_ids[1:seq_len - 1]]
            results.append({"probs": prob_one, "labels": label_one})
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
