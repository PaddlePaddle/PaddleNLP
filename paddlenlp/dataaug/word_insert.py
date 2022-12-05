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
import os
from typing import Iterable

import paddle

from ..transformers import AutoModelForMaskedLM, AutoTokenizer
from .base_augment import BaseAugment

__all__ = ["WordInsert"]


class WordInsert(BaseAugment):
    """
    WordInsert is a word-level insert data augmentation strategy.

    Args:
        aug_type (str or list(str)):
            Insert dictionary type
        custom_file_path (str, optional):
            Custom insert dictionary file path
        delete_file_path (str, optional):
            Dictionary file path for deleting words in insert dictionary
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

    def __init__(
        self,
        aug_type,
        custom_file_path=None,
        delete_file_path=None,
        create_n=1,
        aug_n=None,
        aug_percent=0.02,
        aug_min=1,
        aug_max=10,
    ):
        super().__init__(create_n=create_n, aug_n=aug_n, aug_percent=aug_percent, aug_min=aug_min, aug_max=aug_max)

        self.custom_file_path = custom_file_path
        self.delete_file_path = delete_file_path
        self.model_name = "ernie-1.0"
        if isinstance(aug_type, str):
            self.type = aug_type
            if aug_type in ["synonym", "homonym", "custom"]:
                self.dict = self._load_insert_dict(aug_type)
            elif aug_type in ["mlm"]:
                self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
                self.mlm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif isinstance(aug_type, Iterable):
            self.type = "combination"
            self.dict = {}
            # Merge dictionaries from different sources
            for t in aug_type:
                if t in ["synonym", "homonym", "custom"]:
                    t_dict = self._load_insert_dict(t)
                    for k in t_dict:
                        if k in self.dict:
                            self.dict[k] = list(set(self.dict[k] + t_dict[k]))
                        else:
                            self.dict[k] = t_dict[k]
            # Todo: delete some words in the dictionary
        else:
            self.type = aug_type

    def _load_insert_dict(self, source_type):
        """Load insert dictionary"""
        if source_type in ["synonym", "homonym"]:
            fullname = self._load_file("word_" + source_type)
        elif source_type in ["custom"]:
            fullname = self.custom_file_path
        elif source_type in ["delete"]:
            fullname = self.delete_file_path
        if os.path.exists(fullname):
            with open(fullname, "r", encoding="utf-8") as f:
                insert_dict = json.load(f)
            f.close()
        else:
            raise ValueError("The {} should exist.".format(fullname))
        return insert_dict

    def _augment(self, sequence):
        seq_tokens = self.tokenizer.cut(sequence)
        aug_indexes = self._skip_stop_word_tokens(seq_tokens)
        aug_n = self._get_aug_n(len(seq_tokens), len(aug_indexes))
        if aug_n == 0:
            return []
        elif self.type == "mlm":
            return self._augment_mlm(sequence, seq_tokens, aug_indexes)
        elif aug_n == 1:
            return self._augment_single(seq_tokens, aug_indexes)
        else:
            return self._augment_multi(seq_tokens, aug_n, aug_indexes)

    @paddle.no_grad()
    def _augment_mlm(self, sequence, seq_tokens, aug_indexes):

        t = 0
        sentences = []
        while t < self.create_n * self.loop and len(sentences) < self.create_n:
            skip = False
            t += 1
            p = random.randint(0, 1)
            idx = random.sample(aug_indexes, 1)[0]
            aug_tokens = [[idx, "[MASK]" * len(seq_tokens[idx])]]
            sequence_mask = self._generate_sequence(seq_tokens.copy(), aug_tokens, p)
            tokenized = self.mlm_tokenizer(sequence_mask)
            masked_positions = [
                i for i, idx in enumerate(tokenized["input_ids"]) if idx == self.mlm_tokenizer.mask_token_id
            ]
            output = self.mlm_model(
                paddle.to_tensor([tokenized["input_ids"]]), paddle.to_tensor([tokenized["token_type_ids"]])
            )
            predicted = "".join(
                self.mlm_tokenizer.convert_ids_to_tokens(paddle.argmax(output[0][masked_positions], axis=-1))
            )
            for p in predicted:
                if p in self.stop_words:
                    skip = True
                    break
            if skip:
                continue

            aug_tokens = [[idx, predicted]]

            sequence_generate = self._generate_sequence(seq_tokens.copy(), aug_tokens, p)
            if sequence_generate != sequence and sequence_generate not in sentences:
                sentences.append(sequence_generate)
        return sentences

    def _augment_multi(self, seq_tokens, aug_n, aug_indexes):
        sentences = []
        if self.type in ["synonym", "homonym", "combination", "custom"]:
            candidate_tokens = []
            for aug_index in aug_indexes:
                if seq_tokens[aug_index] in self.dict:
                    candidate_tokens.append([aug_index, self.dict[seq_tokens[aug_index]]])
            aug_n = min(aug_n, len(candidate_tokens))
            if aug_n != 0:
                t = 0
                while t < self.create_n * self.loop and len(sentences) < self.create_n:
                    t += 1
                    idxes = random.sample(list(range(len(candidate_tokens))), aug_n)
                    aug_tokens = []
                    for idx in idxes:
                        aug_index, aug_dict = candidate_tokens[idx]
                        aug_tokens.append([aug_index, random.sample(aug_dict, 1)[0]])
                    p = random.randint(0, 1)
                    sentence = self._generate_sequence(seq_tokens.copy(), aug_tokens, p)
                    if sentence not in sentences:
                        sentences.append(sentence)
        elif self.type in ["random"]:
            t = 0
            while t < self.create_n * self.loop and len(sentences) < self.create_n:
                t += 1
                aug_tokens = []
                aug_indexes = random.sample(aug_indexes, aug_n)
                for aug_index in aug_indexes:
                    token = self.vocab.to_tokens(random.randint(0, len(self.vocab) - 2))
                    aug_tokens.append([aug_index, token])
                p = random.randint(0, 1)
                sentence = self._generate_sequence(seq_tokens.copy(), aug_tokens, p)
                if sentence not in sentences:
                    sentences.append(sentence)
        return sentences

    def _augment_single(self, seq_tokens, aug_indexes):

        sentences = []
        aug_tokens = []
        if self.type in ["synonym", "homonym", "combination", "custom"]:
            candidate_tokens = []
            for aug_index in aug_indexes:
                if seq_tokens[aug_index] in self.dict:
                    for token in self.dict[seq_tokens[aug_index]]:
                        candidate_tokens.append([aug_index, token])
            create_n = min(self.create_n, len(candidate_tokens))
            aug_tokens = random.sample(candidate_tokens, create_n)
        elif self.type in ["random"]:
            t = 0
            while t < self.create_n * self.loop and len(aug_tokens) < self.create_n:
                t += 1
                aug_index = random.sample(aug_indexes, 1)[0]
                token = self.vocab.to_tokens(random.randint(0, len(self.vocab) - 2))
                if [aug_index, token] not in aug_tokens:
                    aug_tokens.append([aug_index, token])
        for aug_token in aug_tokens:
            p = random.randint(0, 1)
            sentences.append(self._generate_sequence(seq_tokens.copy(), [aug_token], p))
        return sentences

    def _generate_sequence(self, output_seq_tokens, aug_tokens, p):
        """Genearte the sequences according to the mapping list"""
        for aug_token in aug_tokens:
            idx, token = aug_token
            if p == 0:
                output_seq_tokens[idx] = token + output_seq_tokens[idx]
            else:
                output_seq_tokens[idx] += token
        return "".join(output_seq_tokens)
