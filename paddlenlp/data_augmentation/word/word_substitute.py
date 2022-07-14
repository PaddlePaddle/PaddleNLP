# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import json
from typing import Iterable

sys.path.append("..")
from base_augment import BaseAugment


class WordSubstitute(BaseAugment):
    """
    WordSubstitute is a word-level substitution data augmentation strategy
    that supports replacing words in the input sequence based on existing
    dictionaries or custom dictionaries. In the future, WordSubstitute will
    support selecting alternative words based on TF-IDF and generating words
    based on pretrained model MLM tasks.

    Args:
        aug_type (str or list(str)):
            Substitution dictionary type
        custom_file_path (str, optional):
            Custom substitution dictionary file path
        delete_file_path (str, optional):
            Dictionary file path for deleting words in substitution dictionary
        create_n (int):
            Number of augmented sequences.
        aug_n (int):
            Number of augmented words in sequences.
        aug_percent (int):
            Percentage of augmented words in sequences.
        aug_min (int):
            Minimum number of augmented words in sequences.
        aug_max (int):
            Maximum number of augmented words in sequences.
        tf_idf (bool):
            Use tf-idf to select the most unimportant word for substitution.
    """

    def __init__(self,
                 aug_type,
                 custom_file_path=None,
                 delete_file_path=None,
                 create_n=1,
                 aug_n=None,
                 aug_percent=None,
                 aug_min=1,
                 aug_max=10,
                 tf_idf=False):
        super().__init__(create_n=create_n,
                         aug_n=aug_n,
                         aug_percent=aug_percent,
                         aug_min=aug_min,
                         aug_max=aug_max)

        self.custom_file_path = custom_file_path
        self.delete_file_path = delete_file_path
        self.tf_idf = tf_idf

        if isinstance(aug_type, str):
            self.type = aug_type
            if aug_type in ['synonym', 'homonym', 'custom']:
                self.dict = self._load_substitue_dict(aug_type)
        elif isinstance(aug_type, Iterable):
            self.type = 'combination'
            self.dict = {}
            # Merge dictionaries from different sources
            for t in aug_type:
                if t in ['synonym', 'homonym', 'custom']:
                    t_dict = self._load_substitue_dict(t)
                    for k in t_dict:
                        if k in self.dict:
                            self.dict[k] = list(set(self.dict[k] + t_dict[k]))
                        else:
                            self.dict[k] = t_dict[k]
            # Todo: delete some words in the dictionary
        else:
            self.type = aug_type

    def _load_substitue_dict(self, source_type):
        '''Load substitution dictionary'''
        if source_type in ['synonym', 'homonym']:
            fullname = self._load_file('word_' + source_type)
        elif source_type in ['custom']:
            fullname = self.custom_file_path
        elif source_type in ['delete']:
            fullname = self.delete_file_path
        with open(fullname, 'r', encoding='utf-8') as f:
            substitue_dict = json.load(f)
        f.close()
        return substitue_dict

    def _reverse_sequence(self, output_seq_tokens, aug_tokens):
        '''Genearte the sequences according to the mapping list'''
        for aug_token in aug_tokens:
            idx, token = aug_token
            output_seq_tokens[idx] = token
        return ''.join(output_seq_tokens)

    def _augment(self, sequence):
        if self.type == 'mlm':
            self.aug_n = 1
            return self._augment_mlm(sequence)
        seq_tokens = self.tokenizer.cut(sequence)
        aug_n = self._get_aug_n(len(seq_tokens))
        if aug_n == 1:
            return self._augment_single(seq_tokens)
        else:
            return self._augment_multi(seq_tokens, aug_n)

    def _augment_mlm(self, sequence):
        # Todo: generate word based on mlm task
        raise NotImplementedError

    def _augment_multi(self, seq_tokens, aug_n):
        sentences = []
        aug_indexes = self._skip_stop_word_tokens(seq_tokens)
        aug_n = min(aug_n, len(aug_indexes))
        if self.type in ['synonym', 'homonym', 'combination', 'custom']:
            candidate_tokens = []
            for aug_index in aug_indexes:
                if seq_tokens[aug_index] in self.dict:
                    candidate_tokens.append(
                        [aug_index, self.dict[seq_tokens[aug_index]]])
            aug_n = min(aug_n, len(candidate_tokens))
            if aug_n != 0:
                t = 0
                while t < self.create_n * self.loop and len(
                        sentences) < self.create_n:
                    t += 1
                    idxes = random.sample(list(range(len(candidate_tokens))),
                                          aug_n)
                    aug_tokens = []
                    for idx in idxes:
                        aug_index, aug_dict = candidate_tokens[idx]
                        aug_tokens.append(
                            [aug_index,
                             random.sample(aug_dict, 1)[0]])

                    sentence = self._reverse_sequence(seq_tokens.copy(),
                                                      aug_tokens)
                    if sentence not in sentences:
                        sentences.append(sentence)
        elif self.type in ['random']:
            t = 0
            while t < self.create_n * self.loop and len(
                    sentences) < self.create_n:
                t += 1
                aug_tokens = []
                aug_indexes = random.sample(aug_indexes, aug_n)
                for aug_index in aug_indexes:
                    token = self.vocab.to_tokens(
                        random.randint(0,
                                       len(self.vocab) - 2))
                    aug_tokens.append([aug_index, token])
                sentence = self._reverse_sequence(seq_tokens.copy(), aug_tokens)
                if sentence not in sentences:
                    sentences.append(sentence)
        return sentences

    def _augment_single(self, seq_tokens):
        aug_indexes = self._skip_stop_word_tokens(seq_tokens)
        sentences = []
        aug_tokens = []
        if self.type in ['synonym', 'homonym', 'combination', 'custom']:
            candidate_tokens = []
            for aug_index in aug_indexes:
                if seq_tokens[aug_index] in self.dict:
                    for token in self.dict[seq_tokens[aug_index]]:
                        candidate_tokens.append([aug_index, token])
            create_n = min(self.create_n, len(candidate_tokens))
            aug_tokens = random.sample(candidate_tokens, create_n)
        elif self.type in ['random']:
            t = 0
            while t < self.create_n * self.loop and len(
                    aug_tokens) < self.create_n:
                t += 1
                aug_index = random.sample(aug_indexes, 1)[0]
                token = self.vocab.to_tokens(
                    random.randint(0,
                                   len(self.vocab) - 2))
                if [aug_index, token] not in aug_tokens:
                    aug_tokens.append([aug_index, token])
        for aug_token in aug_tokens:
            sentences.append(
                self._reverse_sequence(seq_tokens.copy(), [aug_token]))
        return sentences


if __name__ == '__main__':
    aug = WordSubstitute(['synonym', 'homonym'], create_n=2, aug_n=3)
    s1 = '2021年，我再看深度学习领域，无论是自然语言处理、音频信号处理、图像处理、推荐系统，似乎都看到attention混得风生水起，只不过更多时候看到的是它的另一个代号：Transformer。'
    s2 = '绝对准确率计算的是完全预测正确的样本占总样本数的比例，而0-1损失计算的是完全预测错误的样本占总样本的比例。'
    augmented_list = aug.augment([s1, s2])
    for augmented in augmented_list:
        for a in augmented:
            print(a)
        print('---------------------')
    aug = WordSubstitute('synonym', create_n=2, aug_percent=3)
    augmented_list = aug.augment(s1)
    for a in augmented:
        print(a)
