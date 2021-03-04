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

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.embeddings import TokenEmbedding


def parse_decodes(ds, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_word = dict(zip(ds.word_vocab.values(), ds.word_vocab.keys()))
    id_label = dict(zip(ds.label_vocab.values(), ds.label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = [id_word[x] for x in ds.word_ids[idx][:end]]
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
                sub_word_ids = convert_tokens_to_ids(words, self.word_vocab,
                                                     'OOV')
                sub_label_ids = convert_tokens_to_ids(labels, self.label_vocab,
                                                      'O')
                self.word_ids.append(sub_word_ids)
                self.label_ids.append(sub_label_ids)
        self.word_num = max(self.word_vocab.values()) + 1
        self.label_num = max(self.label_vocab.values()) + 1

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], len(self.word_ids[index]), self.label_ids[
            index]


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
                extended_vocab_path='./conf/word.dic', unknown_token='OOV')
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

    train_ds = ExpressDataset('./data/train.txt')
    dev_ds = ExpressDataset('./data/dev.txt')
    test_ds = ExpressDataset('./data/test.txt')

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=train_ds.word_vocab.get('OOV')),
        Stack(),
        Pad(axis=0, pad_val=train_ds.label_vocab.get('O'))
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

    network = BiGRUWithCRF(300, 300, train_ds.word_num, train_ds.label_num)
    model = paddle.Model(network)

    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters())
    crf_loss = LinearChainCrfLoss(network.crf)
    chunk_evaluator = ChunkEvaluator(
        label_list=train_ds.label_vocab.keys(), suffix=True)
    model.prepare(optimizer, crf_loss, chunk_evaluator)

    model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=10,
              save_dir='./results',
              log_freq=1)

    model.evaluate(eval_data=test_loader)
    outputs, lens, decodes = model.predict(test_data=test_loader)
    preds = parse_decodes(test_ds, decodes, lens)

    file_path = "bigru_results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
    # Print some examples
    print(
        "The results have been saved in the file: %s, some examples are shown below: "
        % file_path)
    print("\n".join(preds[:10]))
