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
import unittest
import paddle
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.utils.log import logger
from utils import get_vocab_list
logger.logger.setLevel('ERROR')


class TestTokenEmbedding(unittest.TestCase):
    def setUp(self):
        self.config = {}
        self.set_config()
        self.embedding = TokenEmbedding(**self.config)

    def set_config(self):
        # default config
        self.config[
            "embedding_name"] = "w2v.sikuquanshu.target.word-word.dim300"
        self.config["trainable"] = True


class TestTokenEmbeddingUNK(TestTokenEmbedding):
    def set_config(self):
        super().set_config()
        self.config["unknown_token"] = "[unk]"  # default [UNK], change it
        self.config["unknown_token_vector"] = np.random.normal(
            scale=0.02, size=300).astype(paddle.get_default_dtype())

    def test_unk_token(self):
        self.assertEqual(self.config["unknown_token"],
                         self.embedding.unknown_token)
        self.assertTrue(
            np.allclose(self.config["unknown_token_vector"],
                        self.embedding.search(self.embedding.unknown_token)))


class TestTokenEmbeddingExtendedVocab(TestTokenEmbedding):
    def set_config(self):
        super().set_config()
        self.config["extended_vocab_path"] = "./data/dict.txt"

    def test_extended_vocab(self):
        vocab_list = get_vocab_list(self.config["extended_vocab_path"])
        emb_idx = set(self.embedding.get_idx_list_from_words(vocab_list))
        vocab_idx = set([i for i in range(len(vocab_list))])
        self.assertEqual(emb_idx, vocab_idx)


class TestTokenEmbeddingKeepExtendedVocab(TestTokenEmbedding):
    def set_config(self):
        super().set_config()
        self.config["extended_vocab_path"] = "./data/dict.txt"
        self.config["keep_extended_vocab_only"] = True

    def test_extended_vocab(self):
        vocab_list = get_vocab_list(self.config["extended_vocab_path"])
        vocab_size = len(vocab_list)
        # +1 means considering [PAD]
        self.assertEqual(vocab_size + 1, len(self.embedding._word_to_idx))


class TestTokenEmbeddingSimilarity(TestTokenEmbedding):
    def set_config(self):
        super().set_config()
        self.config["extended_vocab_path"] = "./data/dict.txt"
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
        vocab_list = get_vocab_list(self.config["extended_vocab_path"])
        word_a, word_b, vec_a, vec_b = self.get_random_word_vec(vocab_list)
        result = self.embedding.cosine_sim(word_a, word_b)
        expected_result = self.get_cosine(vec_a, vec_b)
        self.assertTrue(np.allclose(result, expected_result))

    def test_dot(self):
        vocab_list = get_vocab_list(self.config["extended_vocab_path"])
        word_a, word_b, vec_a, vec_b = self.get_random_word_vec(vocab_list)
        result = self.embedding.dot(word_a, word_b)
        expected_result = self.get_dot(vec_a, vec_b)
        self.assertTrue(np.allclose(result, expected_result))


if __name__ == "__main__":
    unittest.main()
