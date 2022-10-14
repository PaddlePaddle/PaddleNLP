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

import os
import time

import argparse
import numpy as np

import paddle
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad
# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_file", type=str, required=True, default='./static_graph_params.pdmodel', help="The path to model info in static graph.")
parser.add_argument("--params_file", type=str, required=True, default='./static_graph_params.pdiparams', help="The path to parameters in static graph.")
parser.add_argument("--data_dir", type=str, default=None, help="The folder where the dataset is located.")
parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init model.")
parser.add_argument("--batch_size", type=int, default=2, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=64, help="Number of words of the longest seqence.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument("--epochs", default=1, type=int, help="The number of epochs when running benchmark.")

args = parser.parse_args()
# yapf: enable


def normalize_token(token, normlize_vocab):
    """Normalize text from DBC case to SBC case"""
    if normlize_vocab:
        token = normlize_vocab.get(token, token)
    return token


def convert_tokens_to_ids(tokens,
                          vocab,
                          oov_replace_token=None,
                          normlize_vocab=None):
    """Convert tokens to token indexs"""
    token_ids = []
    oov_replace_token = vocab.get(
        oov_replace_token) if oov_replace_token else None
    for token in tokens:
        token = normalize_token(token, normlize_vocab)
        token_id = vocab.get(token, oov_replace_token)
        token_ids.append(token_id)

    return token_ids


def convert_example(tokens, max_seq_len, word_vocab, normlize_vocab=None):
    """Convert tokens of sequences to token ids"""
    tokens = tokens[:max_seq_len]

    token_ids = convert_tokens_to_ids(tokens,
                                      word_vocab,
                                      oov_replace_token="OOV",
                                      normlize_vocab=normlize_vocab)
    length = len(token_ids)
    return token_ids, length


def load_vocab(dict_path):
    """Load vocab from file"""
    vocab = {}
    reverse = None
    with open(dict_path, "r", encoding='utf8') as fin:
        for i, line in enumerate(fin):
            terms = line.strip("\n").split("\t")
            if len(terms) == 2:
                if reverse == None:
                    reverse = True if terms[0].isdigit() else False
                if reverse:
                    value, key = terms
                else:
                    key, value = terms
            elif len(terms) == 1:
                key, value = terms[0], i
            else:
                raise ValueError("Error line: %s in file: %s" %
                                 (line, dict_path))
            vocab[key] = value
    return vocab


def parse_result(words, preds, lengths, word_vocab, label_vocab):
    """ Parse padding result """
    batch_out = []
    id2word_dict = dict(zip(word_vocab.values(), word_vocab.keys()))
    id2label_dict = dict(zip(label_vocab.values(), label_vocab.keys()))
    for sent_index in range(len(lengths)):
        sent = [
            id2word_dict[index]
            for index in words[sent_index][:lengths[sent_index]]
        ]
        tags = [
            id2label_dict[index]
            for index in preds[sent_index][:lengths[sent_index]]
        ]

        sent_out = []
        tags_out = []
        parital_word = ""
        for ind, tag in enumerate(tags):
            # for the first word
            if parital_word == "":
                parital_word = sent[ind]
                tags_out.append(tag.split('-')[0])
                continue

            # for the beginning of word
            if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                sent_out.append(parital_word)
                tags_out.append(tag.split('-')[0])
                parital_word = sent[ind]
                continue

            parital_word += sent[ind]

        # append the last word, except for len(tags)=0
        if len(sent_out) < len(tags_out):
            sent_out.append(parital_word)

        batch_out.append([sent_out, tags_out])
    return batch_out


class Predictor(object):

    def __init__(self, model_file, params_file, device, max_seq_length):
        self.max_seq_length = max_seq_length

        config = paddle.inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]

        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])

    def predict(self,
                data,
                word_vocab,
                label_vocab,
                normlize_vocab,
                batch_size=1):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
                A Example object contains `text`(word_ids) and `seq_len`(sequence length).
            word_vocab(obj:`dict`): The word id (key) to word str (value) map.
            label_vocab(obj:`dict`): The label id (key) to label str (value) map.
            normlize_vocab(obj:`dict`): The fullwidth char (key) to halfwidth char (value) map.
            batch_size(obj:`int`, defaults to 1): The number of batch.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        examples = []

        for text in data:
            tokens = list(text.strip())
            token_ids, length = convert_example(tokens,
                                                self.max_seq_length,
                                                word_vocab=word_vocab,
                                                normlize_vocab=normlize_vocab)
            examples.append((token_ids, length))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=0),  # input
            Stack(axis=0),  # length
        ): fn(samples)

        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]

        results = []

        for batch in batches:
            token_ids, length = batchify_fn(batch)
            self.input_handles[0].copy_from_cpu(token_ids)
            self.input_handles[1].copy_from_cpu(length)
            self.predictor.run()
            preds = self.output_handle.copy_to_cpu()
            result = parse_result(token_ids, preds, length, word_vocab,
                                  label_vocab)
            results.extend(result)
        return results


if __name__ == "__main__":
    word_vocab = load_vocab(os.path.join(args.data_dir, 'word.dic'))
    label_vocab = load_vocab(os.path.join(args.data_dir, 'tag.dic'))
    normlize_vocab = load_vocab(os.path.join(args.data_dir, 'q2b.dic'))
    infer_ds = []
    with open(os.path.join(args.data_dir, 'infer.tsv'), "r",
              encoding="utf-8") as fp:
        for line in fp.readlines():
            infer_ds += [line.strip()]
    predictor = Predictor(args.model_file, args.params_file, args.device,
                          args.max_seq_len)
    start = time.time()
    for _ in range(args.epochs):
        results = predictor.predict(infer_ds,
                                    word_vocab,
                                    label_vocab,
                                    normlize_vocab,
                                    batch_size=args.batch_size)
    end = time.time()
    for idx, result in enumerate(results):
        print('Text: {}'.format(infer_ds[idx]))
        sent_tags = []
        sent, tags = result
        sent_tag = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print('Result: {}\n'.format(sent_tag))
    print("Total predict time: {:.4f} s".format(end - start))
