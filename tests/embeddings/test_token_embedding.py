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
import numpy as np
import os
import unittest
import paddle
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.utils.log import logger
from util import get_vocab_list, create_test_data

from common_test import CommonTest
logger.logger.setLevel('ERROR')


class TestTokenEmbedding(CommonTest):
    def setUp(self):
        self.test_data_file = create_test_data(__file__)
        self.config[
            "embedding_name"] = "w2v.sikuquanshu.target.word-word.dim300"
        self.config["trainable"] = True


class TestTokenEmbeddingTrainable(TestTokenEmbedding):
    def test_trainable(self):
        self.embedding = TokenEmbedding(**self.config)
        self.check_output_not_equal(self.config["trainable"],
                                    self.embedding.weight.stop_gradient)


class TestTokenEmbeddingUNK(TestTokenEmbedding):
    def setUp(self):
        super().setUp()
        self.config["unknown_token"] = "[unk]"  # default [UNK], change it
        self.config["unknown_token_vector"] = np.random.normal(
            scale=0.02, size=300).astype(paddle.get_default_dtype())

    def test_unk_token(self):
        self.embedding = TokenEmbedding(**self.config)
        self.check_output_equal(self.config["unknown_token"],
                                self.embedding.unknown_token)
        self.check_output_equal(
            self.config["unknown_token_vector"],
            self.embedding.search(self.embedding.unknown_token)[0])


class TestTokenEmbeddingExtendedVocab(TestTokenEmbedding):
    def setUp(self):
        super().setUp()
        self.config["extended_vocab_path"] = self.test_data_file

    def test_extended_vocab(self):
        self.embedding = TokenEmbedding(**self.config)
        vocab_list = get_vocab_list(self.config["extended_vocab_path"])
        emb_idx = set(self.embedding.get_idx_list_from_words(vocab_list))
        vocab_idx = set([i for i in range(len(vocab_list))])
        self.assertEqual(emb_idx, vocab_idx)
        self.check_output_equal(emb_idx, vocab_idx)


class TestTokenEmbeddingKeepExtendedVocab(TestTokenEmbedding):
    def setUp(self):
        super().setUp()
        self.config["extended_vocab_path"] = self.test_data_file
        self.config["keep_extended_vocab_only"] = True

    def test_extended_vocab(self):
        self.embedding = TokenEmbedding(**self.config)
        vocab_list = get_vocab_list(self.config["extended_vocab_path"])
        vocab_size = len(vocab_list)
        # +1 means considering [PAD]
        self.check_output_equal(vocab_size + 1,
                                len(self.embedding._word_to_idx))


class TestTokenEmbeddingSimilarity(TestTokenEmbedding):
    def setUp(self):
        super().setUp()
        self.config["extended_vocab_path"] = self.test_data_file
        self.config["keep_extended_vocab_only"] = True

    def get_dot(self, vec_a, vec_b):
        return np.sum(vec_a * vec_b)

    def get_cosine(self, vec_a, vec_b):
        return self.get_dot(vec_a, vec_b) / (np.sqrt(
            self.get_dot(vec_a, vec_a) * self.get_dot(vec_b, vec_b)))

    def get_random_word_vec(self, vocab_list):
        vocab_size = len(vocab_list)
        ids = np.random.randint(vocab_size, size=2)
        word_a, word_b = vocab_list[ids[0]], vocab_list[ids[1]]
        vec_a, vec_b = self.embedding.search([word_a, word_b])
        return word_a, word_b, vec_a, vec_b

    def test_cosine_sim(self):
        self.embedding = TokenEmbedding(**self.config)
        vocab_list = get_vocab_list(self.config["extended_vocab_path"])
        word_a, word_b, vec_a, vec_b = self.get_random_word_vec(vocab_list)
        result = self.embedding.cosine_sim(word_a, word_b)
        expected_result = self.get_cosine(vec_a, vec_b)
        self.check_output_equal(result, expected_result)

    def test_dot(self):
        self.embedding = TokenEmbedding(**self.config)
        vocab_list = get_vocab_list(self.config["extended_vocab_path"])
        word_a, word_b, vec_a, vec_b = self.get_random_word_vec(vocab_list)
        result = self.embedding.dot(word_a, word_b)
        expected_result = self.get_dot(vec_a, vec_b)
        self.check_output_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
