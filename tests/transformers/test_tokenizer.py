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
from multiprocessing import Process
from tempfile import TemporaryDirectory
from parameterized import parameterized

from paddle import nn
from paddlenlp.transformers.model_utils import PretrainedModel, MODEL_HOME
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from paddlenlp.transformers import BertTokenizer, BasicTokenizer, WordpieceTokenizer

from paddlenlp.transformers.bert.modeling import BertForPretraining
from paddlenlp.transformers.gpt.modeling import GPTForPretraining
from paddlenlp.transformers.tinybert.modeling import TinyBertForPretraining

from paddlenlp.transformers.bert.tokenizer import BertTokenizer
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer, GPTChineseTokenizer
from paddlenlp.transformers.tinybert.tokenizer import TinyBertTokenizer

from tests.common_test import CpuCommonTest, CommonTest
from tests.util import slow, assert_raises


class FakePretrainedModel(PretrainedModel):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, *args, **kwargs):
        pass


def get_pretrained_models_params() -> List[Tuple[str, Type[PretrainedModel]]]:
    """get all of pretrained model names in some PretrainedModels

    Returns:
        List[Tuple[str, Type[PretrainedModel]]]: the parameters of unit test method
    """
    model_types: List[PretrainedModel] = [
        BertForPretraining, GPTForPretraining, TinyBertForPretraining
    ]
    name_class_tuples: List[Tuple[str, Type[PretrainedModel]]] = []
    for ModelType in model_types:
        for model_name in ModelType.pretrained_resource_files_map.get(
                'model_state', {}).keys():
            name_class_tuples.append([model_name, ModelType])
    return name_class_tuples


def get_pretrained_tokenzier_params(
) -> List[Tuple[str, Type[PretrainedTokenizer]]]:
    """get all of pretrained tokenzier names in some PretrainedTokenzier

    Returns:
        List[Tuple[str, Type[PretrainedTokenzier]]]: the parameters of unit test method
    """
    tokenizer_types: List[PretrainedTokenizer] = [
        BertTokenizer, GPTTokenizer, GPTChineseTokenizer, TinyBertTokenizer
    ]
    name_class_params: List[Tuple[str, Type[PretrainedTokenizer]]] = []
    for TokenizerType in tokenizer_types:
        for model_name in TokenizerType.pretrained_resource_files_map.get(
                'vocab_file', {}).keys():
            name_class_params.append([model_name, TokenizerType])
    return name_class_params


class TestPretrainedFromPretrained(CpuCommonTest):
    """module for test pretrained model"""

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        model = FakePretrainedModel()
        model.save_pretrained(self.temp_dir.name)

    def do_pretrained_in_process(self):
        FakePretrainedModel.from_pretrained(self.temp_dir.name)

    @parameterized.expand([(1, ), (8, ), (20, ), (50, ), (100, ), (1000, )])
    def test_model_config_writing(self, process_num: int):
        for _ in range(process_num):
            process = Process(target=self.do_pretrained_in_process)
            process.start()

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

        # TODO(wj-Mcat): make this test code pass
        # from pretrained from the dir
        # PretrainedModelClass.from_pretrained(cache_dir)

        # remove the cache model file
        shutil.rmtree(cache_dir, ignore_errors=True)

    @parameterized.expand(get_pretrained_tokenzier_params())
    def test_pretrained_tokenizer(
            self, tokenizer_name: str,
            PretrainedTokenzierClass: Type[PretrainedTokenizer]):
        """stupid test on the pretrained tokenzier"""
        cache_dir = os.path.join(MODEL_HOME, tokenizer_name)
        shutil.rmtree(cache_dir, ignore_errors=True)

        tokenizer: PretrainedTokenzierClass = PretrainedTokenzierClass.from_pretrained(
            tokenizer_name)

        files = os.listdir(cache_dir)
        self.assertTrue(
            os.path.exists(
                os.path.join(cache_dir, tokenizer.tokenizer_config_file)))
        for resource_file_name in tokenizer.resource_files_names.values():
            self.assertTrue(
                os.path.exists(os.path.join(cache_dir, resource_file_name)))

        # TODO(wj-Mcat): make this test code pass
        # from_pretrained from the dir
        # PretrainedTokenzierClass.from_pretrained(cache_dir)

        # remove the cache model file
        shutil.rmtree(cache_dir, ignore_errors=True)
