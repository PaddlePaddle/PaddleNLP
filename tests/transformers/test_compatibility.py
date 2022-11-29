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

from typing import List
from unittest import TestCase
from paddlenlp.cli.main import load_all_models
from paddlenlp.transformers import BertPretrainedModel, GPTPretrainedModel, AlbertPretrainedModel, RobertaPretrainedModel


class TestCompatibility(TestCase):

    def test_model_compatibility(self):
        """test compatibility of model
            TODO(wj-Mcat): add unittest in `forward`
        """
        pass

    def get_model_names(self) -> List[str]:
        model_names = []
        model_names.extend(
            list(BertPretrainedModel.pretrained_init_configuration.keys()))
        model_names.extend(
            list(GPTPretrainedModel.pretrained_init_configuration.keys()))
        model_names.extend(
            list(AlbertPretrainedModel.pretrained_init_configuration.keys()))
        model_names.extend(
            list(RobertaPretrainedModel.pretrained_init_configuration.keys()))

        return model_names

    def test_tokenizer_compatibility(self):
        """test compatibility of tokenizer between paddlenlp and transformer"""
        for model_name in self.get_model_names():
            self.run_compatibility_test(model_name)

    def run_compatibility_test(self, model_name: str):
        from transformers import AutoTokenizer
        from transformers.tokenization_utils import PreTrainedTokenizer
        hf_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name)

        from paddlenlp.transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 1. test token_id
        assert len(hf_tokenizer) == len(tokenizer)
        for token_id in range(len(tokenizer)):
            hf_token = hf_tokenizer.convert_ids_to_tokens(token_id)
            token = tokenizer.convert_ids_to_tokens(token_id)

            assert hf_token == token

        # 2. convert id to tokens
        import paddle
        random_ids = paddle.randint(high=len(tokenizer), shape=[100]).tolist()
        hf_tokens = hf_tokenizer.convert_ids_to_tokens(random_ids)
        tokens = tokenizer.convert_ids_to_tokens(random_ids)

        for index in range(100):
            assert hf_tokens[index] == tokens[index]

        # 3. convert tokens to string
        hf_string = hf_tokenizer.convert_tokens_to_string(hf_tokens)
        string = tokenizer.convert_tokens_to_string(tokens)

        assert hf_string == string

        # 4. convert string to tokens
        hf_tokens = hf_tokenizer.tokenize(hf_string)
        tokens = tokenizer.tokenize(string)
        assert len(hf_tokens) == len(tokens)

        for index in range(len(tokens)):
            assert hf_tokens[index] == tokens[index]

        # 5. convert tokens to ids
        hf_ids = hf_tokenizer.convert_tokens_to_ids(hf_tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        assert len(hf_ids) == len(ids)

        for index in range(len(ids)):
            assert hf_ids[index] == ids[index]
