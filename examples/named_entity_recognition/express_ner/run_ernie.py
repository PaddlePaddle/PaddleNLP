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
from functools import partial

import paddle
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErniePretrainedModel, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator


def parse_decodes(ds, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(ds.label_vocab.values(), ds.label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.word_ids[idx][:end]
        tags = [id_label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval precision: %f - recall: %f - f1: %f" %
          (precision, recall, f1_score))


def predict(model, data_loader, ds):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds = parse_decodes(ds, pred_list, len_list)
    return preds


def convert_example(example, tokenizer, label_vocab):
    tokens, labels = example
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(tokens)
    lens = len(input_ids)
    labels = ['O'] + labels + ['O']
    labels = [label_vocab[x] for x in labels]
    return input_ids, segment_ids, lens, labels


def load_dict(dict_path):
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab


class ExpressDataset(paddle.io.Dataset):
    def __init__(self, data_path):
        self.word_vocab = load_dict('./conf/word.dic')
        self.label_vocab = load_dict('./conf/tag.dic')
        self.word_ids = []
        self.label_ids = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                self.word_ids.append(words)
                self.label_ids.append(labels)
        self.word_num = max(self.word_vocab.values()) + 1
        self.label_num = max(self.label_vocab.values()) + 1

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index]


if __name__ == '__main__':
    paddle.set_device('gpu')

    train_ds = ExpressDataset('./data/train.txt')
    dev_ds = ExpressDataset('./data/dev.txt')
    test_ds = ExpressDataset('./data/test.txt')

    tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
    trans_func = partial(
        convert_example, tokenizer=tokenizer, label_vocab=train_ds.label_vocab)

    ignore_label = -1
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Stack(),
        Pad(axis=0, pad_val=ignore_label)
    ): fn(list(map(trans_func, samples)))

    train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=200,
        shuffle=True,
        return_list=True,
        collate_fn=batchify_fn)
    dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=200,
        return_list=True,
        collate_fn=batchify_fn)
    test_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_size=200,
        return_list=True,
        collate_fn=batchify_fn)

    model = ErnieForTokenClassification.from_pretrained(
        "ernie-1.0", num_classes=train_ds.label_num)

    metric = ChunkEvaluator(label_list=train_ds.label_vocab.keys(), suffix=True)
    loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=2e-5, parameters=model.parameters())

    step = 0
    for epoch in range(10):
        model.train()
        for idx, (input_ids, segment_ids, length,
                  labels) in enumerate(train_loader):
            logits = model(input_ids, segment_ids).reshape(
                [-1, train_ds.label_num])
            loss = paddle.mean(loss_fn(logits, labels.reshape([-1])))
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()
            step += 1
            print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
        evaluate(model, metric, dev_loader)

        paddle.save(model.state_dict(),
                    './ernie_result/model_%d.pdparams' % step)

    preds = predict(model, test_loader, test_ds)
    print('\n'.join(preds[:10]))
