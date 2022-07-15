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

from paddlenlp.data_augmentation import BaseAugment


class WordDelete(BaseAugment):
    """
    WordDelete is a word-level deletion data augmentation strategy.

    Args:
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
    """

    def __init__(self,
                 create_n=1,
                 aug_n=None,
                 aug_percent=None,
                 aug_min=1,
                 aug_max=10):
        super().__init__(create_n=create_n,
                         aug_n=aug_n,
                         aug_percent=aug_percent,
                         aug_min=aug_min,
                         aug_max=aug_max)

    def _augment(self, sequence):

        seq_tokens = self.tokenizer.cut(sequence)
        aug_n = self._get_aug_n(len(seq_tokens))
        aug_indexes = self.skip_words(seq_tokens)
        aug_n = min(aug_n, len(aug_indexes))
        t = 0
        sentences = []
        while t < self.create_n * self.loop and len(sentences) < self.create_n:
            t += 1
            idxes = random.sample(aug_indexes, aug_n)
            sentence = ''
            for idx in range(len(seq_tokens)):
                if idx not in idxes:
                    sentence += seq_tokens[idx]
            if sentence not in sentences:
                sentences.append(sentence)
        return sentences

    def skip_words(self, seq_tokens):
        '''Skip words. We can rewrite function to skip specify words.'''
        indexes = []
        for i, seq_token in enumerate(seq_tokens):
            if seq_token not in self.stop_words and not seq_token.isdigit(
            ) and not seq_token.encode('UTF-8').isalpha():
                indexes.append(i)
        return indexes


if __name__ == '__main__':
    aug = WordDelete(create_n=10, aug_n=1)
    s1 = '2021年，我再看深度学习领域，无论是自然语言处理、音频信号处理、图像处理、推荐系统，似乎都看到attention混得风生水起，只不过更多时候看到的是它的另一个代号：Transformer。'

    augmented = aug.augment(s1)
    print(s1)
    for a in augmented:
        print(a)
