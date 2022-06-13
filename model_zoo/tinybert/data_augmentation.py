# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 Huawei Technologies Co., Ltd.
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

import random
import sys
import os
import unicodedata
import re
import logging
import csv
import argparse

import numpy as np
import paddle

from paddlenlp.transformers import BertTokenizer, BertForPretraining

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

StopWordsList = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't", "'s", "'re"
]


def strip_accents(text):
    """
    Strip accents from input String.
    :param text: The input string.
    :type text: String.
    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError):
        # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


# valid string only includes al
def _is_valid(string):
    return True if not re.search('[^a-z]', string) else False


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def prepare_embedding_retrieval(glove_file, vocab_size=100000):
    cnt = 0
    words = []
    embeddings = {}

    # only read first 100,000 words for fast retrieval
    with open(glove_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split()
            words.append(items[0])
            embeddings[items[0]] = [float(x) for x in items[1:]]

            cnt += 1
            if cnt == vocab_size:
                break

    vocab = {w: idx for idx, w in enumerate(words)}
    ids_to_tokens = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(embeddings[ids_to_tokens[0]])
    emb_matrix = np.zeros((vocab_size, vector_dim))
    for word, v in embeddings.items():
        if word == '<unk>':
            continue
        emb_matrix[vocab[word], :] = v

    # normalize each word vector
    d = (np.sum(emb_matrix**2, 1)**0.5)
    emb_norm = (emb_matrix.T / d).T
    return emb_norm, vocab, ids_to_tokens


class DataAugmentor(object):

    def __init__(self, model, tokenizer, emb_norm, vocab, ids_to_tokens, M, N,
                 p):
        self.model = model
        self.tokenizer = tokenizer
        self.emb_norm = emb_norm
        self.vocab = vocab
        self.ids_to_tokens = ids_to_tokens
        self.M = M
        self.N = N
        self.p = p

    def _word_distance(self, word):
        if word not in self.vocab.keys():
            return []
        word_idx = self.vocab[word]
        word_emb = self.emb_norm[word_idx]

        dist = np.dot(self.emb_norm, word_emb.T)
        dist[word_idx] = -np.Inf

        candidate_ids = np.argsort(-dist)[:self.M]
        return [self.ids_to_tokens[idx] for idx in candidate_ids][:self.M]

    def _masked_language_model(self, sent, word_pieces, mask_id):
        tokenized_text = self.tokenizer.tokenize(sent)
        tokenized_text = ['[CLS]'] + tokenized_text
        tokenized_len = len(tokenized_text)

        tokenized_text = word_pieces + ['[SEP]'
                                        ] + tokenized_text[1:] + ['[SEP]']

        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[:512]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * (tokenized_len + 1) + [1] * (len(tokenized_text) -
                                                          tokenized_len - 1)

        tokens_tensor = paddle.to_tensor([token_ids])
        segments_tensor = paddle.to_tensor([segments_ids])

        predictions, _ = self.model(tokens_tensor, segments_tensor)
        word_candidates = paddle.argsort(
            predictions[0, mask_id], descending=True)[:self.M].numpy().tolist()
        word_candidates = self.tokenizer.convert_ids_to_tokens(word_candidates)

        return list(filter(lambda x: x.find("##"), word_candidates))

    def _word_augment(self, sentence, mask_token_idx, mask_token):
        word_pieces = self.tokenizer.tokenize(sentence)
        word_pieces = ['[CLS]'] + word_pieces
        tokenized_len = len(word_pieces)

        token_idx = -1
        for i in range(1, tokenized_len):
            if "##" not in word_pieces[i]:
                token_idx = token_idx + 1
                if token_idx < mask_token_idx:
                    word_piece_ids = []
                elif token_idx == mask_token_idx:
                    word_piece_ids = [i]
                else:
                    break
            else:
                word_piece_ids.append(i)

        if len(word_piece_ids) == 1:
            word_pieces[word_piece_ids[0]] = '[MASK]'
            candidate_words = self._masked_language_model(
                sentence, word_pieces, word_piece_ids[0])
        elif len(word_piece_ids) > 1:
            candidate_words = self._word_distance(mask_token)
        else:
            logger.info("invalid input sentence!")

        if len(candidate_words) == 0:
            candidate_words.append(mask_token)

        return candidate_words

    def augment(self, sent):
        candidate_sents = [sent]

        tokens = self.tokenizer.basic_tokenizer.tokenize(sent)
        candidate_words = {}
        for (idx, word) in enumerate(tokens):
            if _is_valid(word) and word not in StopWordsList:
                candidate_words[idx] = self._word_augment(sent, idx, word)
        logger.info(candidate_words)
        cnt = 0
        while cnt < self.N:
            new_sent = list(tokens)
            for idx in candidate_words.keys():
                candidate_word = random.choice(candidate_words[idx])

                x = random.random()
                if x < self.p:
                    new_sent[idx] = candidate_word

            if " ".join(new_sent) not in candidate_sents:
                candidate_sents.append(' '.join(new_sent))
            cnt += 1

        return candidate_sents


class AugmentProcessor(object):

    def __init__(self, augmentor, glue_dir, task_name):
        self.augmentor = augmentor
        self.glue_dir = glue_dir
        self.task_name = task_name
        self.augment_ids = {
            'MRPC': [3, 4],
            'MNLI': [8, 9],
            'CoLA': [3],
            'SST-2': [0],
            'STS-B': [7, 8],
            'QQP': [3, 4],
            'QNLI': [1, 2],
            'RTE': [1, 2]
        }

        self.filter_flags = {
            'MRPC': True,
            'MNLI': True,
            'CoLA': False,
            'SST-2': True,
            'STS-B': True,
            'QQP': True,
            'QNLI': True,
            'RTE': True
        }

        assert self.task_name in self.augment_ids

    def read_augment_write(self):
        task_dir = os.path.join(self.glue_dir, self.task_name)
        train_samples = _read_tsv(os.path.join(task_dir, "train.tsv"))
        output_filename = os.path.join(task_dir, "train_aug.tsv")

        augment_ids_ = self.augment_ids[self.task_name]
        filter_flag = self.filter_flags[self.task_name]

        with open(output_filename, 'w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for (i, line) in enumerate(train_samples):
                if i == 0 and filter_flag:
                    writer.writerow(line)
                    continue

                for augment_id in augment_ids_:
                    sent = line[augment_id]
                    augmented_sents = self.augmentor.augment(sent)
                    for augment_sent in augmented_sents:
                        line[augment_id] = augment_sent
                        writer.writerow(line)

                if (i + 1) % 1000 == 0:
                    logger.info("Having been processing {} examples".format(
                        str(i + 1)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_bert_model",
        default=None,
        type=str,
        required=True,
        help=
        "Downloaded pretrained model (bert-base-uncased) is under this folder")
    parser.add_argument("--glove_embs",
                        default=None,
                        type=str,
                        required=True,
                        help="Glove word embeddings file")
    parser.add_argument("--glue_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="GLUE data dir")
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help=
        "Task(eg. CoLA, SST-2) that we want to do data augmentation for its train set"
    )
    parser.add_argument("--N",
                        default=30,
                        type=int,
                        help="How many times is the corpus expanded?")
    parser.add_argument(
        "--M",
        default=15,
        type=int,
        help="Choose from M most-likely words in the corresponding position")
    parser.add_argument("--p",
                        default=0.4,
                        type=float,
                        help="Threshold probability p to replace current word")

    parser.add_argument("--device",
                        default="gpu",
                        type=str,
                        help="device, gpu or cpu")

    args = parser.parse_args()
    logger.info(args)

    default_params = {
        "CoLA": {
            "N": 30
        },
        "MNLI": {
            "N": 10
        },
        "MRPC": {
            "N": 30
        },
        "SST-2": {
            "N": 20
        },
        "STS-b": {
            "N": 30
        },
        "QQP": {
            "N": 10
        },
        "QNLI": {
            "N": 20
        },
        "RTE": {
            "N": 30
        }
    }

    device = paddle.set_device(args.device)
    if args.task_name in default_params:
        args.N = default_params[args.task_name]["N"]

    # Prepare data augmentor
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)
    model = BertForPretraining.from_pretrained(args.pretrained_bert_model)
    model.eval()

    emb_norm, vocab, ids_to_tokens = prepare_embedding_retrieval(
        args.glove_embs)

    data_augmentor = DataAugmentor(model, tokenizer, emb_norm, vocab,
                                   ids_to_tokens, args.M, args.N, args.p)

    # Do data augmentation
    processor = AugmentProcessor(data_augmentor, args.glue_dir, args.task_name)
    processor.read_augment_write()


if __name__ == "__main__":
    main()
