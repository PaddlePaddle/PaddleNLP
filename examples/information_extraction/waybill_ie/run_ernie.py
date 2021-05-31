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

from functools import partial

import paddle
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator

from data import load_dict, load_dataset, parse_decodes


def convert_to_features(example, tokenizer, label_vocab):
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens, return_length=True, is_split_into_words=True)
    # Token '[CLS]' and '[SEP]' will get label 'O'
    labels = ['O'] + labels + ['O']
    tokenized_input['labels'] = [label_vocab[x] for x in labels]
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("[EVAL] Precision: %f - Recall: %f - F1: %f" %
          (precision, recall, f1_score))
    model.train()


@paddle.no_grad()
def predict(model, data_loader, ds, label_vocab):
    all_preds = []
    all_lens = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1)
        # Drop CLS prediction
        preds = [pred[1:] for pred in preds.numpy()]
        all_preds.append(preds)
        all_lens.append(lens)
    sentences = [example[0] for example in ds.data]
    results = parse_decodes(sentences, all_preds, all_lens, label_vocab)
    return results


if __name__ == '__main__':
    paddle.set_device('gpu')

    # Create dataset, tokenizer and dataloader.
    train_ds, dev_ds, test_ds = load_dataset(datafiles=(
        './data/train.txt', './data/dev.txt', './data/test.txt'))

    label_vocab = load_dict('./data/tag.dic')
    tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

    trans_func = partial(
        convert_to_features, tokenizer=tokenizer, label_vocab=label_vocab)

    train_ds.map(trans_func)
    dev_ds.map(trans_func)
    test_ds.map(trans_func)

    ignore_label = -1
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # token_type_ids
        Stack(dtype='int64'),  # seq_len
        Pad(axis=0, pad_val=ignore_label, dtype='int64')  # labels
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
        for input_ids, token_type_ids, length, labels in train_loader:
            logits = model(input_ids, token_type_ids)
            loss = paddle.mean(loss_fn(logits, labels))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            step += 1
            print("[TRAIN] Epoch:%d - Step:%d - Loss: %f" % (epoch, step, loss))
        evaluate(model, metric, dev_loader)

        paddle.save(model.state_dict(), './ernie_ckpt/model_%d.pdparams' % step)

    preds = predict(model, test_loader, test_ds, label_vocab)
    file_path = "ernie_results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
    # Print some examples
    print(
        "The results have been saved in the file: %s, some examples are shown below: "
        % file_path)
    print("\n".join(preds[:10]))
