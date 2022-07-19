#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# !/usr/bin/env python3
"""
Functions for explaining text classifiers.
"""
import itertools
import json
import re
import time
import math
import paddle
from functools import partial

import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state

import LIME.explanation as explanation
import LIME.lime_base as lime_base


class TextDomainMapper(explanation.DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, indexed_string, language):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.indexed_string = indexed_string
        self.language = language

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            exp = [('%s_%s' % (self.indexed_string.word(x[0]), '-'.join(
                map(str, self.indexed_string.string_position(x[0])))), x[1])
                   for x in exp]
        else:
            exp = [(self.indexed_string.word(x[0]), x[1]) for x in exp]
        return exp

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                text=True,
                                opacity=True):
        """Adds text with highlighted words to visualization.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        if not text:
            return u''
        text = (self.indexed_string.raw_string().encode(
            'utf-8', 'xmlcharrefreplace').decode('utf-8'))
        text = re.sub(r'[<>&]', '|', text)
        exp = [(self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]), x[1]) for x in exp]
        all_occurrences = list(
            itertools.chain.from_iterable(
                [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
        ret = '''
            %s.show_raw_text(%s, %d, %s, %s, %s);
            ''' % (exp_object_name, json.dumps(all_occurrences), label,
                   json.dumps(text), div_name, json.dumps(opacity))
        return ret


class IndexedString(object):
    """String with various indexes."""

    def __init__(self,
                 raw_string,
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 language='en'):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace words with this if bow=False
                if None, default value is UNKWORDZ
        """
        self.raw = raw_string
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string
        self.language = language

        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens = set(tokens)

            def non_word(string):
                return string not in tokens

        else:
            # with the split_expression as a non-capturing group (?:), we don't need to filter out
            # the separator character from the split results.
            # splitter = re.compile(r'(%s)|$' % split_expression)
            # self.as_list = [s for s in splitter.split(self.raw) if s]
            if self.language == "ch":
                splitter = re.compile(r'([\u4e00-\u9fa5])')
                self.as_list = [
                    w for w in splitter.split(self.raw) if len(w.strip()) > 0
                ]
            else:
                splitter = re.compile(split_expression)
                self.as_list = [
                    w for w in self.raw.strip().split() if len(w.strip()) > 0
                ]
            valid_word = splitter.match

        self.as_np = np.array(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if (valid_word(word)
                    and self.language == 'en') or (not valid_word(word)
                                                   and self.language == 'ch'):
                non_vocab.add(word)
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if self.language == "ch":
            if not self.bow:
                return ''.join([
                    self.as_list[i] if mask[i] else self.mask_string
                    for i in range(mask.shape[0])
                ])
            return ''.join([self.as_list[v] for v in mask.nonzero()[0]])
        else:
            if not self.bow:
                return ' '.join([
                    self.as_list[i] if mask[i] else self.mask_string
                    for i in range(mask.shape[0])
                ])
            return ' '.join([self.as_list[v] for v in mask.nonzero()[0]])

    @staticmethod
    def _segment_with_tokens(text, tokens):
        """Segment a string around the tokens created by a passed-in tokenizer"""
        list_form = []
        text_ptr = 0
        for token in tokens:
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError(
                        "Tokenization produced tokens that do not belong in string!"
                    )
            text_ptr += len(token)
            if inter_token_string:
                list_form.append(''.join(inter_token_string))
            list_form.append(token)
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])
        return list_form

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(
                itertools.chain.from_iterable(
                    [self.positions[z] for z in words]))
        else:
            return self.positions[words]


class IndexedCharacters(object):
    """String with various indexes."""

    def __init__(self, raw_string, bow=True, mask_string=None):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            bow: if True, a char is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same character. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace characters with this if bow=False
                if None, default value is chr(0)
        """
        self.raw = raw_string
        self.as_list = list(self.raw)
        self.as_np = np.array(self.as_list)
        self.mask_string = chr(0) if mask_string is None else mask_string
        self.string_start = np.arange(len(self.raw))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, char in enumerate(self.as_np):
            if char in non_vocab:
                continue
            if bow:
                if char not in vocab:
                    vocab[char] = len(vocab)
                    self.inverse_vocab.append(char)
                    self.positions.append([])
                idx_char = vocab[char]
                self.positions[idx_char].append(i)
            else:
                self.inverse_vocab.append(char)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join([
                self.as_list[i] if mask[i] else self.mask_string
                for i in range(mask.shape[0])
            ])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(
                itertools.chain.from_iterable(
                    [self.positions[z] for z in words]))
        else:
            return self.positions[words]


class LimeTextExplainer(object):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False,
                 language='en'):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True (bag of words), will perturb input data by removing
                all occurrences of individual words or characters.
                Explanations will be in terms of these words. Otherwise, will
                explain in terms of word-positions, so that a word may be
                important the first time it appears and unimportant the second.
                Only set to false if the classifier uses word order in some way
                (bigrams, etc), or if you set char_level=True.
            mask_string: String used to mask tokens or characters if bow=False
                if None, will be 'UNKWORDZ' if char_level=False, chr(0)
                otherwise.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            char_level: an boolean identifying that we treat each character
                as an independent occurence in the string
        """

        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn,
                                       verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level
        self.language = language

    def explain_instance(self,
                         text_instance: str,
                         tokenizer,
                         pred_label: int,
                         classifier_fn,
                         labels=(0, 1),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None,
                         if_lstm=False):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            text_instance: raw text string to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        indexed_string = (IndexedCharacters(
            text_instance, bow=self.bow, mask_string=self.mask_string)
                          if self.char_level else IndexedString(
                              text_instance,
                              bow=self.bow,
                              split_expression=self.split_expression,
                              mask_string=self.mask_string,
                              language=self.language))
        domain_mapper = TextDomainMapper(indexed_string, self.language)

        # 产生扰动数据集    第一条是原始数据
        # data: 解释器训练特征  list (num_samples, doc_size)
        # yss:  解释器训练标签  list (num_samples, class_num(2))
        # distances: 扰动样本到原始样本的距离 np.array(float) (num_samples, )
        data, yss, distances = self.__data_labels_distances(
            indexed_string,
            tokenizer,
            classifier_fn,
            num_samples,
            distance_metric=distance_metric,
            if_lstm=if_lstm)

        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()

        num_features = indexed_string.num_words()  # 特征数量跟word_num相同

        (ret_exp.intercept[pred_label], ret_exp.local_exp[pred_label],
         ret_exp.score[pred_label], ret_exp.local_pred[pred_label],
         relative_err, err) = self.base.explain_instance_with_data(
             data,
             yss,
             distances,
             pred_label,
             num_features,
             model_regressor=model_regressor,
             feature_selection=self.feature_selection)

        return ret_exp, indexed_string, relative_err, err

    def __data_labels_distances(self,
                                indexed_string,
                                tokenizer,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine',
                                if_lstm=False):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_string: document (IndexedString) to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.

        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        doc_size = indexed_string.num_words()

        if doc_size > 1:
            sample = self.random_state.randint(
                1, doc_size, num_samples -
                1)  # sample: [int(1 ~ doc_size-1) * num_samples-1]
        else:
            sample = [0 for i in range(num_samples - 1)]
        data = np.ones((num_samples, doc_size))
        data[0] = np.ones(doc_size)
        features_range = range(doc_size)
        perturb_text = [indexed_string.raw_string()]  # [文本 * num_samples]

        for i, size in enumerate(sample, start=1):
            # inactive: 从range（0， doc_size）中随机取出的size个数组成的list, 要去掉的字的id
            inactive = self.random_state.choice(
                features_range,  # [0, doc_size)
                size,  # int: 该扰动样本中remove token的数量
                replace=False)

            text = indexed_string.inverse_removing(
                inactive)  # 原文本去掉了inactive中的字后的文本

            data[i, inactive] = 0
            perturb_text.append(text)

        prev_time = time.time()
        # inverse_data: 扰动数据集 [扰动样本 str] * num_samples
        labels = []
        token_ids_list, s_ids_list, seq_len_list = [], [], []
        token_ids_max_len = 0

        valid_idxs = []

        for idx, text in enumerate(perturb_text):
            if self.language == 'en':
                if if_lstm:
                    pad_id = [tokenizer.vocab.token_to_idx.get('[PAD]', 0)]

                    token_ids = tokenizer.encode(text)
                    token_ids_max_len = max(token_ids_max_len, len(token_ids))
                    seq_len = len(token_ids)
                    if seq_len == 0:
                        continue
                    else:
                        valid_idxs.append(idx)
                    seq_len_list.append(seq_len)
                    pad_id = [tokenizer.vocab.token_to_idx.get('[PAD]', 0)]

                else:
                    pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])

                    tokens = tokenizer.tokenize(text)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    token_ids = tokenizer.convert_tokens_to_ids(['[CLS]']) \
                        + token_ids + tokenizer.convert_tokens_to_ids(['[SEP]'])
                    token_ids_max_len = max(token_ids_max_len, len(token_ids))

                token_ids_list.append(token_ids)
            else:
                if len(text) == 0:  # TODO
                    text = perturb_text[0]
                tokens = tokenizer.tokenize(text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                if if_lstm:
                    seq_len = len(token_ids)
                    if seq_len == 0:
                        continue
                    else:
                        valid_idxs.append(idx)
                    seq_len_list.append(seq_len)
                else:
                    token_ids = tokenizer.convert_tokens_to_ids(['[CLS]']) \
                        + token_ids + tokenizer.convert_tokens_to_ids(['[SEP]'])

                # padding
                token_ids = token_ids + tokenizer.convert_tokens_to_ids(
                    ['[PAD]']) * (len(perturb_text[0]) + 2 - len(token_ids))
                token_ids_list.append(token_ids)
                s_ids = [0 for _ in range(len(token_ids))]
                s_ids_list.append(s_ids)

        if self.language == 'en':
            for token_ids in token_ids_list:
                while len(token_ids) < token_ids_max_len:
                    token_ids += pad_id

                s_ids = [0 for _ in range(len(token_ids))]
                s_ids_list.append(s_ids)

        token_ids_np = np.array(token_ids_list)
        s_ids_np = np.array(s_ids_list)
        seq_len_np = np.array(seq_len_list)

        prev_time = time.time()

        batch = 0
        if self.language == "ch":
            length = len(perturb_text[0])

            if if_lstm:
                batch = 128
            else:
                batch = 64 if length < 130 else 50
        else:
            batch = 32

        epoch_num = math.ceil(len(token_ids_np) / batch)
        for idx in range(epoch_num):
            token_ids_tensor = paddle.Tensor(
                value=token_ids_np[idx * batch:(idx + 1) * batch],
                place=paddle.CUDAPlace(0),
                stop_gradient=True)
            if if_lstm:
                seq_len_tensor = paddle.Tensor(
                    value=seq_len_np[idx * batch:(idx + 1) * batch],
                    place=token_ids_tensor.place,
                    stop_gradient=token_ids_tensor.stop_gradient)
                label = classifier_fn(
                    token_ids_tensor,
                    seq_len_tensor)[0]  # label: Tensor[num_samples, 2]
            else:
                s_ids_tensor = paddle.Tensor(
                    value=s_ids_np[idx * batch:(idx + 1) * batch],
                    place=token_ids_tensor.place,
                    stop_gradient=token_ids_tensor.stop_gradient)
                label = classifier_fn(
                    token_ids_tensor,
                    s_ids_tensor)[0]  # label: Tensor[num_samples, 2]

            labels.extend(label.numpy().tolist())

        labels = np.array(labels)  # labels: nsp.array(num_samples, 2)

        print('mode forward time: %.5f' % (time.time() - prev_time))

        distances = distance_fn(sp.sparse.csr_matrix(data))

        return data, labels, distances
