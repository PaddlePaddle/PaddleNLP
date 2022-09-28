# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
from functools import partial

import paddle
import paddle.nn as nn

from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.embeddings import TokenEmbedding

from data import load_dict, load_dataset, parse_decodes
from model import BiGRUWithCRF

parser = argparse.ArgumentParser()

# yapf: disable
parser.add_argument("--save_dir", default='./bigru_crf_ckpt', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument("--data_dir", default='./waybill_ie/data', type=str, help="The folder where the dataset is located.")

args = parser.parse_args()
# yapf: enable


def convert_tokens_to_ids(tokens, vocab, oov_token=None):
    token_ids = []
    oov_id = vocab.get(oov_token) if oov_token else None
    for token in tokens:
        token_id = vocab.get(token, oov_id)
        token_ids.append(token_id)
    return token_ids


def convert_to_features(example, word_vocab, label_vocab):
    tokens, labels = example
    token_ids = convert_tokens_to_ids(tokens, word_vocab, 'OOV')
    label_ids = convert_tokens_to_ids(labels, label_vocab, 'O')
    return token_ids, len(token_ids), label_ids


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for token_ids, lengths, label_ids in data_loader:
        preds = model(token_ids, lengths)
        n_infer, n_label, n_correct = metric.compute(lengths, preds, label_ids)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("[EVAL] Precision: %f - Recall: %f - F1: %f" %
          (precision, recall, f1_score))
    model.train()


@paddle.no_grad()
def predict(model, data_loader, ds, label_vocab):
    all_preds = []
    all_lens = []
    for token_ids, lengths, label_ids in data_loader:
        preds = model(token_ids, lengths)
        all_preds.append(preds.numpy())
        all_lens.append(lengths)
    sentences = [example[0] for example in ds.data]
    results = parse_decodes(sentences, all_preds, all_lens, label_vocab)
    return results


if __name__ == '__main__':
    paddle.set_device(args.device)

    # Create dataset, tokenizer and dataloader.
    train_ds, dev_ds, test_ds = load_dataset(
        datafiles=(os.path.join(args.data_dir, 'train.txt'),
                   os.path.join(args.data_dir, 'dev.txt'),
                   os.path.join(args.data_dir, 'test.txt')))

    label_vocab = load_dict(os.path.join(args.data_dir, 'tag.dic'))
    word_vocab = load_dict(os.path.join(args.data_dir, 'word.dic'))

    trans_func = partial(convert_to_features,
                         word_vocab=word_vocab,
                         label_vocab=label_vocab)
    train_ds.map(trans_func)
    dev_ds.map(trans_func)
    test_ds.map(trans_func)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=word_vocab.get('OOV', 0), dtype='int32'
            ),  # token_ids
        Stack(dtype='int64'),  # seq_len
        Pad(axis=0, pad_val=label_vocab.get('O', 0), dtype='int64')  # label_ids
    ): fn(samples)

    train_loader = paddle.io.DataLoader(dataset=train_ds,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        return_list=True,
                                        collate_fn=batchify_fn)

    dev_loader = paddle.io.DataLoader(dataset=dev_ds,
                                      batch_size=args.batch_size,
                                      drop_last=True,
                                      return_list=True,
                                      collate_fn=batchify_fn)

    test_loader = paddle.io.DataLoader(dataset=test_ds,
                                       batch_size=args.batch_size,
                                       drop_last=True,
                                       return_list=True,
                                       collate_fn=batchify_fn)

    # Define the model netword and its loss
    model = BiGRUWithCRF(300, 256, len(word_vocab), len(label_vocab))

    optimizer = paddle.optimizer.Adam(learning_rate=0.001,
                                      parameters=model.parameters())
    metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)

    step = 0
    for epoch in range(args.epochs):
        for token_ids, lengths, label_ids in train_loader:
            loss = model(token_ids, lengths, label_ids)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            step += 1
            print("[TRAIN] Epoch:%d - Step:%d - Loss: %f" % (epoch, step, loss))
        evaluate(model, metric, dev_loader)
        paddle.save(model.state_dict(),
                    os.path.join(args.save_dir, 'model_%d' % step))

    preds = predict(model, test_loader, test_ds, label_vocab)
    file_path = "bigru_crf_results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
    # Print some examples
    print(
        "The results have been saved into: %s, some examples are shown below: "
        % file_path)
    print("\n".join(preds[:10]))
