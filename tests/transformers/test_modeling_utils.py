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
import unittest
import shutil
from tempfile import TemporaryDirectory
from tests.testing_utils import slow
from multiprocessing import Pool
from paddlenlp.transformers import TinyBertModel, BertModel
from paddlenlp.utils.env import MODEL_HOME


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
