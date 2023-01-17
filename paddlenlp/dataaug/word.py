# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import json
import math
import os
import random
from typing import Iterable

import numpy as np
import paddle

from ..transformers import AutoModelForMaskedLM, AutoTokenizer
from .base_augment import BaseAugment

__all__ = ["WordSubstitute", "WordInsert", "WordSwap", "WordDelete"]


class WordSubstitute(BaseAugment):
    """
    WordSubstitute is a word-level substitution data augmentation strategy
    that supports replacing words in the input sequence based on existing
    dictionaries or custom dictionaries.

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
        tf_idf (str):
            File for calculating TF-IDF score.
        model_name (str):
            Model parameter name for MLM prediction task.
    """

    def __init__(
        self,
        aug_type,
        custom_file_path=None,
        delete_file_path=None,
        create_n=1,
        aug_n=None,
        aug_percent=0.1,
        aug_min=1,
        aug_max=10,
        tf_idf=False,
        tf_idf_file=None,
        model_name="ernie-1.0-large-zh-cw",
        vocab="vocab",
    ):
        super().__init__(
            create_n=create_n, aug_n=aug_n, aug_percent=aug_percent, aug_min=aug_min, aug_max=aug_max, vocab=vocab
        )

        self.custom_file_path = custom_file_path
        self.delete_file_path = delete_file_path
        self.tf_idf = tf_idf
        self.model_name = model_name
        if self.tf_idf:
            self._count_idf(tf_idf_file)

        if isinstance(aug_type, str):
            self.type = aug_type
            if aug_type in ["antonym", "embedding", "synonym", "homonym", "custom"]:
                self.dict = self._load_substitue_dict(aug_type)
            elif aug_type in ["mlm"]:
                self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
                self.mlm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif isinstance(aug_type, Iterable):
            if len(aug_type) == 1:
                self.type = aug_type[0]
            else:
                self.type = "combination"
            if self.type in ["mlm"]:
                self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
                self.mlm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.dict = {}
            # Merge dictionaries from different sources
            for t in aug_type:
                if t in ["antonym", "embedding", "synonym", "homonym", "custom"]:
                    t_dict = self._load_substitue_dict(t)
                    for k in t_dict:
                        if k in self.dict:
                            self.dict[k] = list(set(self.dict[k] + t_dict[k]))
                        else:
                            self.dict[k] = t_dict[k]
            # Todo: delete some words in the dictionary
        else:
            self.type = aug_type

    def _count_idf(self, tf_idf_file):
        if os.path.exists(tf_idf_file):
            with open(tf_idf_file, "r", encoding="utf-8") as f:
                self.word_count_dict = {}
                self.text_tf_idf = []
                self.num = 0
                for line in f:
                    self.num += 1
                    self.text_tf_idf.append(line.strip())
                    for word in set(self.tokenizer.cut(line.strip())):
                        if word not in self.word_count_dict:
                            self.word_count_dict[word] = 0
                        self.word_count_dict[word] += 1
            f.close()
        else:
            raise ValueError("The tf_idf_file should exist.")
        return

    def _calculate_tfidf(self, sequence, seq_tokens, aug_indexes):
        if sequence not in self.text_tf_idf:
            self.num += 1
            self.text_tf_idf.append(sequence)
            for word in set(seq_tokens):
                if word not in self.word_count_dict:
                    self.word_count_dict[word] = 0
                self.word_count_dict[word] += 1
        sequence_count = {}
        for index in aug_indexes:
            if seq_tokens[index] in sequence_count:
                sequence_count[seq_tokens[index]] += 1
            else:
                sequence_count[seq_tokens[index]] = 1
        tfidf = []
        for index in aug_indexes:
            tf = sequence_count[seq_tokens[index]] / len(aug_indexes)
            idf = math.log(self.num / self.word_count_dict[seq_tokens[index]])
            tfidf.append(tf * idf)
        return np.array(tfidf)

    def _load_substitue_dict(self, source_type):
        """Load substitution dictionary"""
        if source_type in ["antonym", "embedding", "synonym", "homonym"]:
            fullname = self._load_file("word_" + source_type)
        elif source_type in ["custom"]:
            fullname = self.custom_file_path
        elif source_type in ["delete"]:
            fullname = self.delete_file_path

        if os.path.exists(fullname):
            with open(fullname, "r", encoding="utf-8") as f:
                substitue_dict = json.load(f)
            f.close()
        else:
            raise ValueError("The {} should exist.".format(fullname))

        return substitue_dict

    def _generate_sequence(self, output_seq_tokens, aug_tokens):
        """Genearte the sequences according to the mapping list"""
        for aug_token in aug_tokens:
            idx, token = aug_token
            output_seq_tokens[int(idx)] = token
        return "".join(output_seq_tokens)

    def _augment(self, sequence):
        seq_tokens = self.tokenizer.cut(sequence)
        aug_indexes = self._skip_stop_word_tokens(seq_tokens)
        aug_n = self._get_aug_n(len(seq_tokens), len(aug_indexes))

        if self.tf_idf:
            tfidf = self._calculate_tfidf(sequence, seq_tokens, aug_indexes)
            p = (max(tfidf) + 0.01 - tfidf) / sum(max(tfidf) + 0.01 - tfidf)
        else:
            p = None

        if aug_n == 0:
            return []
        elif self.type == "mlm":
            return self._augment_mlm(sequence, seq_tokens, aug_indexes, p)
        elif aug_n == 1:
            return self._augment_single(seq_tokens, aug_indexes, p)
        else:
            return self._augment_multi(seq_tokens, aug_n, aug_indexes, p)

    @paddle.no_grad()
    def _augment_mlm(self, sequence, seq_tokens, aug_indexes, p):
        t = 0
        sentences = []
        while t < self.create_n * self.loop * 2 and len(sentences) < self.create_n:
            skip = False
            t += 1
            idx = np.random.choice(aug_indexes, replace=False, p=p)

            aug_tokens = [[idx, "[MASK]" * len(seq_tokens[idx])]]
            sequence_mask = self._generate_sequence(seq_tokens.copy(), aug_tokens)
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
            for ppp in predicted:
                if ppp in self.stop_words:
                    skip = True
                    break
            if skip:
                continue
            aug_tokens = [[idx, predicted]]
            sequence_generate = self._generate_sequence(seq_tokens.copy(), aug_tokens)
            if sequence_generate != sequence and sequence_generate not in sentences:
                sentences.append(sequence_generate)
        return sentences

    def _augment_multi(self, seq_tokens, aug_n, aug_indexes, p):
        sentences = []
        aug_n = min(aug_n, len(aug_indexes))
        if self.type in ["antonym", "embedding", "synonym", "homonym", "combination", "custom"]:
            candidate_tokens = []
            pp = []
            for i, aug_index in enumerate(aug_indexes):
                if seq_tokens[aug_index] in self.dict:
                    candidate_tokens.append([aug_index, self.dict[seq_tokens[aug_index]]])
                    if self.tf_idf:
                        pp.append(p[i])
            pp = np.array(pp)
            pp /= sum(pp)
            aug_n = min(aug_n, len(candidate_tokens))
            if aug_n != 0:
                t = 0
                while t < self.create_n * self.loop and len(sentences) < self.create_n:
                    t += 1
                    if self.tf_idf:
                        idxes = np.random.choice(list(range(len(candidate_tokens))), size=aug_n, replace=False, p=pp)
                    else:
                        idxes = random.sample(list(range(len(candidate_tokens))), aug_n)
                    aug_tokens = []
                    for idx in idxes:
                        aug_index, aug_dict = candidate_tokens[idx]
                        aug_tokens.append([aug_index, random.sample(aug_dict, 1)[0]])

                    sentence = self._generate_sequence(seq_tokens.copy(), aug_tokens)
                    if sentence not in sentences:
                        sentences.append(sentence)
        elif self.type in ["random"]:
            t = 0
            while t < self.create_n * self.loop and len(sentences) < self.create_n:
                t += 1
                aug_tokens = []
                aug_choice_indexes = np.random.choice(aug_indexes, size=aug_n, replace=False, p=p)
                for aug_index in aug_choice_indexes:
                    token = self.vocab.to_tokens(random.randint(0, len(self.vocab) - 2))
                    aug_tokens.append([aug_index, token])
                sentence = self._generate_sequence(seq_tokens.copy(), aug_tokens)
                if sentence not in sentences:
                    sentences.append(sentence)
        return sentences

    def _augment_single(self, seq_tokens, aug_indexes, p):
        sentences = []
        aug_tokens = []
        if self.type in ["antonym", "embedding", "synonym", "homonym", "combination", "custom"]:
            candidate_tokens = []
            pp = []
            for i, aug_index in enumerate(aug_indexes):
                if seq_tokens[aug_index] in self.dict:
                    for token in self.dict[seq_tokens[aug_index]]:
                        candidate_tokens.append([aug_index, token])
                        if self.tf_idf:
                            pp.append(p[i] / len(self.dict[seq_tokens[aug_index]]))
            create_n = min(self.create_n, len(candidate_tokens))
            pp = np.array(pp)
            pp /= sum(pp)
            if self.tf_idf:
                candidate_indexes = np.random.choice(range(len(candidate_tokens)), size=create_n, replace=False, p=pp)
                candidate_tokens = np.array(candidate_tokens)
                aug_tokens = candidate_tokens[candidate_indexes]
            else:
                aug_tokens = random.sample(candidate_tokens, create_n)
        elif self.type in ["random"]:
            t = 0
            while t < self.create_n * self.loop and len(aug_tokens) < self.create_n:
                t += 1
                aug_index = np.random.choice(aug_indexes, replace=False, p=p)
                token = self.vocab.to_tokens(random.randint(0, len(self.vocab) - 2))
                if [aug_index, token] not in aug_tokens:
                    aug_tokens.append([aug_index, token])
        for aug_token in aug_tokens:
            sequence_generate = self._generate_sequence(seq_tokens.copy(), [aug_token])
            sentences.append(sequence_generate)

        return sentences


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
        aug_percent=0.1,
        aug_min=1,
        aug_max=10,
        model_name="ernie-1.0-large-zh-cw",
        vocab="vocab",
    ):
        super().__init__(
            create_n=create_n, aug_n=aug_n, aug_percent=aug_percent, aug_min=aug_min, aug_max=aug_max, vocab=vocab
        )

        self.custom_file_path = custom_file_path
        self.delete_file_path = delete_file_path
        self.model_name = model_name
        if isinstance(aug_type, str):
            self.type = aug_type
            if aug_type in ["antonym", "embedding", "synonym", "homonym", "custom"]:
                self.dict = self._load_insert_dict(aug_type)
            elif aug_type in ["mlm"]:
                self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
                self.mlm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif isinstance(aug_type, Iterable):
            self.type = "combination"
            self.dict = {}
            # Merge dictionaries from different sources
            for t in aug_type:
                if t in ["antonym", "embedding", "synonym", "homonym", "custom"]:
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
        if source_type in ["antonym", "embedding", "synonym", "homonym"]:
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
        if self.type in ["antonym", "embedding", "synonym", "homonym", "combination", "custom"]:
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
        if self.type in ["antonym", "embedding", "synonym", "homonym", "combination", "custom"]:
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


class WordSwap(BaseAugment):
    """
    WordSwap is a word-level swap data augmentation strategy.

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

    def __init__(self, create_n=1, aug_n=None, aug_percent=None, aug_min=1, aug_max=10, vocab="vocab"):
        super().__init__(
            create_n=create_n, aug_n=aug_n, aug_percent=0.1, aug_min=aug_min, aug_max=aug_max, vocab=vocab
        )

    def _augment(self, sequence):

        seq_tokens = self.tokenizer.cut(sequence)
        aug_indexes = self._skip_words(seq_tokens)
        aug_n = self._get_aug_n(len(seq_tokens), len(aug_indexes))

        t = 0
        sentences = []

        if aug_n == 0:
            return []
        while t < self.create_n * self.loop and len(sentences) < self.create_n:
            t += 1
            idxes = random.sample(aug_indexes, aug_n)
            output_seq_tokens = seq_tokens.copy()
            for idx in range(len(seq_tokens)):
                if idx in idxes:
                    output_seq_tokens[idx], output_seq_tokens[idx + 1] = (
                        output_seq_tokens[idx + 1],
                        output_seq_tokens[idx],
                    )
            sentence = "".join(output_seq_tokens)
            if sentence not in sentences:
                sentences.append(sentence)
        return sentences

    def _skip_words(self, seq_tokens):
        """Skip specific words."""
        indexes = []
        for i, seq_token in enumerate(seq_tokens[:-1]):
            if (
                seq_token not in self.stop_words
                and not seq_token.isdigit()
                and not seq_token.encode("UTF-8").isalpha()
            ):
                if (
                    seq_tokens[i + 1] not in self.stop_words
                    and not seq_tokens[i + 1].isdigit()
                    and not seq_tokens[i + 1].encode("UTF-8").isalpha()
                ):
                    indexes.append(i)
        return indexes


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

    def __init__(self, create_n=1, aug_n=None, aug_percent=0.1, aug_min=1, aug_max=10, vocab="vocab"):
        super().__init__(
            create_n=create_n, aug_n=aug_n, aug_percent=aug_percent, aug_min=aug_min, aug_max=aug_max, vocab=vocab
        )

    def _augment(self, sequence):

        seq_tokens = self.tokenizer.cut(sequence)
        aug_indexes = self._skip_words(seq_tokens)
        aug_n = self._get_aug_n(len(seq_tokens), len(aug_indexes))

        t = 0
        sentences = []
        if aug_n == 0:
            return sentences
        while t < self.create_n * self.loop and len(sentences) < self.create_n:
            t += 1
            idxes = random.sample(aug_indexes, aug_n)
            sentence = ""
            for idx in range(len(seq_tokens)):
                if idx not in idxes:
                    sentence += seq_tokens[idx]
            if sentence not in sentences:
                sentences.append(sentence)
        return sentences

    def _skip_words(self, seq_tokens):
        """Skip specific words."""
        indexes = []
        for i, seq_token in enumerate(seq_tokens):
            if (
                seq_token not in self.stop_words
                and not seq_token.isdigit()
                and not seq_token.encode("UTF-8").isalpha()
            ):
                indexes.append(i)
        return indexes
