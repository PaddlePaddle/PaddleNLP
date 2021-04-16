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

import paddle
import paddle.nn as nn

from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.embeddings import TokenEmbedding

from data import load_dict, load_dataset, convert_tokens_to_ids, parse_decodes


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
        Pad(axis=0, pad_val=word_vocab.get('OOV'), dtype='int64'),  # token_ids
        Stack(dtype='int64'),  # seq_len
        Pad(axis=0, pad_val=label_vocab.get('O'), dtype='int64')  # label_ids
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
    sentences = [example[0] for example in test_ds.data]
    preds = parse_decodes(sentences, decodes, lens, label_vocab)

    file_path = "bigru_results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
    # Print some examples
    print(
        "The results have been saved in the file: %s, some examples are shown below: "
        % file_path)
    print("\n".join(preds[:10]))
