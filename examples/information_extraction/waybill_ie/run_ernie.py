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
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator


def parse_decodes(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx][0][:end]
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


def predict(model, data_loader, ds, label_vocab):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds = parse_decodes(ds, pred_list, len_list, label_vocab)
    return preds


def convert_example(example, tokenizer, label_vocab):
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens, return_length=True, is_split_into_words=True)
    # Token '[CLS]' and '[SEP]' will get label 'O'
    labels = ['O'] + labels + ['O']
    tokenized_input['labels'] = [label_vocab[x] for x in labels]
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']


def load_dict(dict_path):
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


if __name__ == '__main__':
    paddle.set_device('gpu')

    # Create dataset, tokenizer and dataloader.
    train_ds, dev_ds, test_ds = load_dataset(datafiles=(
        './data/train.txt', './data/dev.txt', './data/test.txt'))

    label_vocab = load_dict('./conf/tag.dic')
    tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

    trans_func = partial(
        convert_example, tokenizer=tokenizer, label_vocab=label_vocab)

    train_ds.map(trans_func)
    dev_ds.map(trans_func)
    test_ds.map(trans_func)

    ignore_label = -1
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(),  # seq_len
        Pad(axis=0, pad_val=ignore_label)  # labels
    ): fn(samples)

    train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=200,
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

    # Define the model netword and its loss
    model = ErnieForTokenClassification.from_pretrained(
        "ernie-1.0", num_classes=len(label_vocab))
    metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
    loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=2e-5, parameters=model.parameters())

    step = 0
    for epoch in range(10):
        # Switch the model to training mode
        model.train()
        for idx, (input_ids, token_type_ids, length,
                  labels) in enumerate(train_loader):
            logits = model(input_ids, token_type_ids)
            loss = paddle.mean(loss_fn(logits, labels))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            step += 1
            print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
        evaluate(model, metric, dev_loader)

        paddle.save(model.state_dict(),
                    './ernie_result/model_%d.pdparams' % step)

    preds = predict(model, test_loader, test_ds, label_vocab)
    file_path = "ernie_results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
    # Print some examples
    print(
        "The results have been saved in the file: %s, some examples are shown below: "
        % file_path)
    print("\n".join(preds[:10]))
