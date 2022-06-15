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
from typing import Type, List, Tuple
import shutil
import unittest
from parameterized import parameterized
from paddlenlp.transformers.model_utils import PretrainedModel, MODEL_HOME
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from paddlenlp.transformers import BertTokenizer, BasicTokenizer, WordpieceTokenizer

from paddlenlp.transformers.bert.modeling import BertForPretraining
from paddlenlp.transformers.gpt.modeling import GPTForPretraining
from paddlenlp.transformers.tinybert.modeling import TinyBertForPretraining

from paddlenlp.transformers.bert.tokenizer import BertTokenizer
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer, GPTChineseTokenizer

from common_test import CpuCommonTest, CommonTest
from util import slow, assert_raises


def get_pretrained_models_params() -> List[Tuple[str, Type[PretrainedModel]]]:
    model_types: List[PretrainedModel] = [BertForPretraining, GPTForPretraining]
    name_class_tuples: List[Tuple[str, Type[PretrainedModel]]] = []
    for ModelType in model_types:
        for model_name in ModelType.pretrained_resource_files_map.get(
                'model_state', {}).keys():
            name_class_tuples.append([model_name, ModelType])
    return name_class_tuples


def get_pretrained_tokenzier_params(
) -> List[Tuple[str, Type[PretrainedTokenizer]]]:
    tokenizer_types: List[PretrainedTokenizer] = [
        BertTokenizer, GPTTokenizer, GPTChineseTokenizer
    ]
    name_class_params: List[Tuple[str, Type[PretrainedTokenizer]]] = []
    for TokenizerType in tokenizer_types:
        for model_name in TokenizerType.pretrained_resource_files_map.get(
                'vocab_file', {}).keys():
            name_class_params.append([model_name, TokenizerType])
    return name_class_params


class TestPretrainedFromPretrained(CpuCommonTest):
    """module for test pretrained model"""

    @parameterized.expand(get_pretrained_models_params())
    def test_pretrained_model(self, model_name: str,
                              PretrainedModelClass: Type[PretrainedModel]):
        """stupid test"""
        cache_dir = os.path.join(MODEL_HOME, model_name)
        shutil.rmtree(cache_dir, ignore_errors=True)

        model: PretrainedModelClass = PretrainedModelClass.from_pretrained(
            model_name)
        self.assertTrue(
            os.path.exists(os.path.join(cache_dir, model.model_config_file)))

    @parameterized.expand(get_pretrained_tokenzier_params())
    def test_pretrained_tokenizer(
            self, tokenizer_name: str,
            PretrainedTokenzierClass: Type[PretrainedTokenizer]):
        cache_dir = os.path.join(MODEL_HOME, tokenizer_name)
        shutil.rmtree(cache_dir, ignore_errors=True)

        tokenizer: PretrainedTokenzierClass = PretrainedTokenzierClass.from_pretrained(
            tokenizer_name)
        self.assertTrue(
            os.path.exists(
                os.path.join(cache_dir, tokenizer.tokenizer_config_file)))
        self.assertTrue(os.path.exists(os.path.join(cache_dir, )))
