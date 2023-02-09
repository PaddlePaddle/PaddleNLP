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
    """
    TestMarkupLM
    """

    def setUp(self):
        self.input_ids = [[2, 2, 4, 5, 6]]
        self.model_name = "microsoft/markuplm-base"
        self.ground_truth = 0.31490570306777954

    def testLoad(self):
        """load torch weight"""
        # import torch;
        # a = torch.load("~/.paddlenlp/models/models--microsoft--markuplm-base/snapshots/7b1ef356189e6803bd5b58b305c783fd1d395ae6/pytorch_model.bin", map_location="cpu");
        # [print(x) for x in sorted(a.keys())];
        # python -c 'import torch;a = torch.load("~/.paddlenlp/models/models--microsoft--markuplm-base/snapshots/7b1ef356189e6803bd5b58b305c783fd1d395ae6/pytorch_model.bin", map_location="cpu");[print(x) for x in sorted(a.keys())];'
        pass

    def testForwardMarkuplmPaddle(self):
        """_summary_"""
        import sys

        import paddle

        sys.path.append("./")
        from markuplm.models.markuplm import MarkupLMModel

        # pad_token_id=1 have diff for nn.Embedding, so 1 should not in input_ids
        input_ids = [[2, 2, 4, 5, 6]]
        model = MarkupLMModel.from_pretrained("microsoft/markuplm-base", from_hf_hub=True)
        model.eval()
        out = model(input_ids=paddle.to_tensor(input_ids, dtype="int64"), return_dict=True)
        result_paddle = out.pooler_output.abs().mean().item()
        np.testing.assert_allclose(result_paddle, 0.31490570306777954, atol=1e-6)

    def testForwardMarkuplmTorch(self):
        """_summary_"""
        import torch
        from markuplmft.models.markuplm import MarkupLMModel

        input_ids = [[2, 2, 4, 5, 6]]

        model = MarkupLMModel.from_pretrained("microsoft/markuplm-base")
        out = model(input_ids=torch.tensor(input_ids, dtype=torch.long))
        result_torch = out.pooler_output.abs().mean().item()
        np.testing.assert_allclose(result_torch, 0.31490570306777954, atol=1e-6)
