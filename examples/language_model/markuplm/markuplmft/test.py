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

import unittest

import numpy as np


class TestMarkupLM(unittest.TestCase):
    def setUp(self):
        # sys.path.remove(".")
        pass

    def testLoad(self):
        """load torch weight"""
        # import torch;a = torch.load("/ssd2/zhonghui03/.paddlenlp/models/models--microsoft--markuplm-base/snapshots/7b1ef356189e6803bd5b58b305c783fd1d395ae6/pytorch_model.bin", map_location="cpu");[print(x) for x in sorted(a.keys())];
        # python -c 'import torch;a = torch.load("/ssd2/zhonghui03/.paddlenlp/models/models--microsoft--markuplm-base/snapshots/7b1ef356189e6803bd5b58b305c783fd1d395ae6/pytorch_model.bin", map_location="cpu");[print(x) for x in sorted(a.keys())];'
        pass

    def testForwardLM(self):
        """ """
        import paddle

        from .models.markuplm import MarkupLMModel

        model = MarkupLMModel.from_pretrained("microsoft/markuplm-base", from_hf_hub=True)
        model.eval()
        # pad_token_id have diff for nn.Embedding
        out = model(input_ids=paddle.to_tensor([[2, 2, 4, 5, 6]], dtype="int64"), return_dict=True)
        # E       ValueError: Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=False,
        # E              [0.32763067])
        result = out.pooler_output.abs().mean().item()
        print(result)
        # print("sasxxx\n"*20)
        np.testing.assert_allclose(result, 0.31490570306777954, atol=1e-6)
