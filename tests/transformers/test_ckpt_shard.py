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

import os
import sys
import tempfile
import unittest

import numpy as np
import paddle

# paddle.set_device("cpu")


class TestCkptShard(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, ".")

    def test_import(self):
        import inspect

        import paddlenlp

        print(inspect.getfile(paddlenlp))

    @unittest.skip("")
    def testTorch(self):
        from transformers import AutoModel

        model = AutoModel.from_pretrained("bert-base-cased")
        # If you save it using save_pretrained(), you will get a new folder with two files: the config of the model and its weights:

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            print(sorted(os.listdir(tmp_dir)))
        # ['config.json', 'pytorch_model.bin']
        # Now let’s use a maximum shard size of 200MB:

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, max_shard_size="200MB")
            print(sorted(os.listdir(tmp_dir)))
        # ['config.json', 'pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin', 'pytorch_model.bin.index.json']

    @unittest.skip("")
    def testPaddleLoad(self):
        from paddlenlp.transformers import AutoModel

        model = AutoModel.from_pretrained("bert-base-cased")
        model.eval()

    @unittest.skip("")
    def testPaddleLoadLocal(self):
        from paddlenlp.transformers import AutoModel, AutoModelForSequenceClassification

        model = AutoModel.from_pretrained("bert-base-cased")
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
            model.save_pretrained(tmp_dir)
            new_model = AutoModel.from_pretrained(tmp_dir)
            new_model.eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            print(sorted(os.listdir(tmp_dir)))
            new_model = AutoModel.from_pretrained(tmp_dir)
            new_model.eval()

    def testPaddleLoadShard(self):
        from paddlenlp.transformers import AutoModel

        model = AutoModel.from_pretrained("bert-base-cased")
        model.eval()
        ret = model(input_ids=paddle.to_tensor([[x for x in range(100, 110)]], dtype="int64"), return_dict=True)
        v1 = ret.last_hidden_state.abs().mean().item()

        # ['config.json', 'pytorch_model.bin']
        # Now let’s use a maximum shard size of 200MB:
        # raise ValueError()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, max_shard_size="200MB", safe_serialization=True)
            print(sorted(os.listdir(tmp_dir)))
            new_model = AutoModel.from_pretrained(tmp_dir)
            # print(new_model.state_dict().keys())
            new_model.eval()
            ret = new_model(
                input_ids=paddle.to_tensor([[x for x in range(100, 110)]], dtype="int64"), return_dict=True
            )
            v2 = ret.last_hidden_state.abs().mean().item()
            np.testing.assert_allclose(v1, v2, rtol=1e-9)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, max_shard_size="200MB")
            print(sorted(os.listdir(tmp_dir)))
            new_model = AutoModel.from_pretrained(tmp_dir)
            # print(new_model.state_dict().keys())
            new_model.eval()
            ret = new_model(
                input_ids=paddle.to_tensor([[x for x in range(100, 110)]], dtype="int64"), return_dict=True
            )
            v3 = ret.last_hidden_state.abs().mean().item()
            np.testing.assert_allclose(v1, v3, rtol=1e-9)

        # ['config.json', 'pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin', 'pytorch_model.bin.index.json']
