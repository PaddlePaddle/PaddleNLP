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

import os
import shutil
import tempfile
import unittest
from multiprocessing import Pool
from tempfile import TemporaryDirectory

import paddle
import pytest

from paddlenlp.transformers import (
    BertModel,
    PretrainedConfig,
    PretrainedModel,
    TinyBertModel,
)
from paddlenlp.transformers.model_utils import dtype_guard, register_base_model
from paddlenlp.utils.env import CONFIG_NAME, MODEL_HOME, PADDLE_WEIGHT_FILE_NAME
from tests.testing_utils import slow


class FakeConfig(PretrainedConfig):
    def __init__(self, use_fp32_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_fp32_norm = True


class FakePretrainedModel(PretrainedModel):
    config_class = FakeConfig

    def keep_in_fp32_modules(cls, key: str, config: FakeConfig, dtype: str) -> bool:
        if config.use_fp32_norm and "norm" in key:
            return True
        return False


@register_base_model
class FakeModel(FakePretrainedModel):
    def __init__(self, config: FakeConfig):
        super(FakeModel, self).__init__(config)
        self.linear = paddle.nn.Linear(2, 3)

        with dtype_guard("float32"):
            self.norm = paddle.nn.LayerNorm(2)


def download_bert_model(model_name: str):
    """set the global method: multiprocessing can not pickle local method

    Args:
        model_name (str): the model name
    """

    model = BertModel.from_pretrained(model_name)
    # free the model resource
    del model


class TestModeling(unittest.TestCase):
    """Test PretrainedModel single time, not in Transformer models"""

    def test_from_pretrained_cache_dir_community_model(self):
        model_name = "__internal_testing__/bert"
        with TemporaryDirectory() as tempdir:
            BertModel.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, PADDLE_WEIGHT_FILE_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))

    @slow
    def test_from_pretrained_cache_dir_pretrained_init(self):
        model_name = "bert-base-uncased"
        with TemporaryDirectory() as tempdir:
            BertModel.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, PADDLE_WEIGHT_FILE_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))

    @slow
    def test_from_pretrained_with_load_as_state_np_params(self):
        """init model with `load_state_as_np` params"""
        model = TinyBertModel.from_pretrained("tinybert-4l-312d", load_state_as_np=True)
        self.assertIsNotNone(model)

    @slow
    def test_multiprocess_downloading(self):
        """test downloading with multi-process. Some errors may be triggered when downloading model
        weight file with multiprocess, so this test code was born.

        `num_process_in_pool` is the number of process in Pool.
        And the `num_jobs` is the number of total process to download file.
        """
        num_process_in_pool, num_jobs = 10, 20
        small_model_path = (
            "https://paddlenlp.bj.bcebos.com/models/community/__internal_testing__/bert/model_state.pdparams"
        )

        from paddlenlp.transformers.model_utils import get_path_from_url_with_filelock

        with TemporaryDirectory() as tempdir:

            with Pool(num_process_in_pool) as pool:
                pool.starmap(get_path_from_url_with_filelock, [(small_model_path, tempdir) for _ in range(num_jobs)])

    @slow
    def test_model_from_pretrained_with_multiprocessing(self):
        """
        this test can not init tooooo many models which will occupy CPU/GPU memorys.

            `num_process_in_pool` is the number of process in Pool.
            And the `num_jobs` is the number of total process to download file.
        """
        num_process_in_pool, num_jobs = 1, 10

        # 1.remove tinybert model weight file
        model_name = "__internal_testing__/bert"
        shutil.rmtree(os.path.join(MODEL_HOME, model_name), ignore_errors=True)

        # 2. downloaing tinybert modeling using multi-processing
        with Pool(num_process_in_pool) as pool:
            pool.starmap(download_bert_model, [(model_name,) for _ in range(num_jobs)])

    def test_keep_in_fp32(self):
        with tempfile.TemporaryDirectory() as tempdir:

            with dtype_guard("float16"):
                config = FakeConfig()
                model = FakeModel(config)
                model.config = config

                model.save_pretrained(tempdir)

                # check model_state.pdparams
                state_dict = paddle.load(os.path.join(tempdir, "model_state.pdparams"))
                self.assertEqual(state_dict["linear.weight"].dtype, paddle.float16)
                self.assertEqual(state_dict["norm.weight"].dtype, paddle.float32)

                # cast all to fp16
                state_dict = {k: paddle.cast(v, "float16") for k, v in state_dict.items()}
                paddle.save(state_dict, os.path.join(tempdir, "model_state.pdparams"))

                model: FakeModel = FakeModel.from_pretrained(tempdir)

                self.assertEqual(model.linear.weight.dtype, paddle.float16)
                self.assertEqual(model.norm.weight.dtype, paddle.float32)

    def test_keep_in_fp32_with_error(self):
        with tempfile.TemporaryDirectory() as tempdir:

            with dtype_guard("float16"):
                config = FakeConfig()
                config.use_fp32_norm = False
                model = FakeModel(config)
                model.config = config

                model.save_pretrained(tempdir)

                # check model_state.pdparams
                state_dict = paddle.load(os.path.join(tempdir, "model_state.pdparams"))
                self.assertEqual(state_dict["linear.weight"].dtype, paddle.float16)
                self.assertEqual(state_dict["norm.weight"].dtype, paddle.float32)

                # cast all to fp16
                state_dict = {k: paddle.cast(v, "float16") for k, v in state_dict.items()}
                paddle.save(state_dict, os.path.join(tempdir, "model_state.pdparams"))

                with pytest.raises(AssertionError, match=r".*Variable dtype not match.*"):
                    model: FakeModel = FakeModel.from_pretrained(tempdir, use_fp32_norm=False)

                model: FakeModel = FakeModel.from_pretrained(tempdir, use_fp32_norm=True)
                self.assertEqual(model.linear.weight.dtype, paddle.float16)
                self.assertEqual(model.norm.weight.dtype, paddle.float32)
