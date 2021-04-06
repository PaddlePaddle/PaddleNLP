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

import paddle
import paddle.nn as nn

from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.embeddings import TokenEmbedding


def parse_decodes(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx][0][:end]
        tags = [id_label[x] for x in decodes[idx][:end]]
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


def convert_tokens_to_ids(tokens, vocab, oov_token=None):
    token_ids = []
    oov_id = vocab.get(oov_token) if oov_token else None
    for token in tokens:
        token_id = vocab.get(token, oov_id)
        token_ids.append(token_id)
    return token_ids


def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        vocab[key] = i
        i += 1
    return vocab


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


class BiGRUWithCRF(nn.Layer):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 word_num,
                 label_num,
                 use_w2v_emb=False):
        super(BiGRUWithCRF, self).__init__()
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(
                extended_vocab_path='./data/word.dic', unknown_token='OOV')
        else:
            self.word_emb = nn.Embedding(word_num, emb_size)
        self.gru = nn.GRU(emb_size,
                          hidden_size,
                          num_layers=2,
                          direction='bidirectional')
        self.fc = nn.Linear(hidden_size * 2, label_num + 2)  # BOS EOS
        self.crf = LinearChainCrf(label_num)
        self.decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, x, lens):
        embs = self.word_emb(x)
        output, _ = self.gru(embs)
        output = self.fc(output)
        _, pred = self.decoder(output, lens)
        return output, lens, pred


if __name__ == '__main__':
    paddle.set_device('gpu')

    # Create dataset, tokenizer and dataloader.
    train_ds, dev_ds, test_ds = load_dataset(datafiles=(
        './data/train.txt', './data/dev.txt', './data/test.txt'))

    label_vocab = load_dict('./data/tag.dic')
    word_vocab = load_dict('./data/word.dic')

    def convert_example(example):
        tokens, labels = example
        token_ids = convert_tokens_to_ids(tokens, word_vocab, 'OOV')
        label_ids = convert_tokens_to_ids(labels, label_vocab, 'O')
        return token_ids, len(token_ids), label_ids

    train_ds.map(convert_example)
    dev_ds.map(convert_example)
    test_ds.map(convert_example)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=word_vocab.get('OOV')),  # token_ids
        Stack(),  # seq_len
        Pad(axis=0, pad_val=label_vocab.get('O'))  # label_ids
    ): fn(samples)

    train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=200,
        shuffle=True,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

    dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=200,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

    test_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_size=200,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

    # Define the model netword and its loss
    network = BiGRUWithCRF(300, 300, len(word_vocab), len(label_vocab))
    model = paddle.Model(network)
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters())
    crf_loss = LinearChainCrfLoss(network.crf)
    chunk_evaluator = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
    model.prepare(optimizer, crf_loss, chunk_evaluator)

    model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=10,
              save_dir='./results',
              log_freq=1)

    model.evaluate(eval_data=test_loader)
    outputs, lens, decodes = model.predict(test_data=test_loader)
    preds = parse_decodes(test_ds, decodes, lens, label_vocab)

    file_path = "bigru_results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
    # Print some examples
    print(
        "The results have been saved in the file: %s, some examples are shown below: "
        % file_path)
    print("\n".join(preds[:10]))
