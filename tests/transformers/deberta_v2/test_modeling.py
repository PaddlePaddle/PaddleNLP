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
from __future__ import annotations

import tempfile
import unittest
from typing import List

import numpy as np
import paddle

from paddlenlp.transformers.deberta_v2.configuration import DebertaV2Config
from paddlenlp.transformers.model_utils import PretrainedModel

from ...testing_utils import require_package


class BertCompatibilityTest(unittest.TestCase):
    test_model_id = "hf-tiny-model-private/tiny-random-DebertaV2Model"

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import DebertaV2Model

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        model = DebertaV2Model.from_pretrained(cls.test_model_id)
        model.save_pretrained(cls.torch_model_path)

    def test_model_config_mapping(self):
        config = DebertaV2Config(num_labels=22, hidden_dropout_prob=0.99)
        self.assertEqual(config.hidden_dropout_prob, 0.99)
        self.assertEqual(config.num_labels, 22)

    def setUp(self) -> None:
        self.tempdirs: List[tempfile.TemporaryDirectory] = []

    def tearDown(self) -> None:
        for tempdir in self.tempdirs:
            tempdir.cleanup()

    def get_tempdir(self) -> str:
        tempdir = tempfile.TemporaryDirectory()
        self.tempdirs.append(tempdir)
        return tempdir.name

    def compare_two_model(self, first_model: PretrainedModel, second_model: PretrainedModel):

        first_weight_name = "encoder.layer.3.attention.self.in_proj.weight"

        second_weight_name = "encoder.layer.3.attention.self.in_proj.weight"

        first_tensor = first_model.state_dict()[first_weight_name]
        second_tensor = second_model.state_dict()[second_weight_name]
        self.compare_two_weight(first_tensor, second_tensor)

    def compare_two_weight(self, first_tensor, second_tensor):
        diff = paddle.sum(first_tensor - second_tensor).numpy().item()
        self.assertEqual(diff, 0.0)

    @require_package("transformers", "torch")
    def test_deberta_converter(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the paddle model
            from paddlenlp.transformers.deberta_v2.modeling import DebertaV2Model

            paddle_model = DebertaV2Model.from_pretrained(
                "hf-internal-testing/tiny-random-DebertaV2Model", from_hf_hub=True, cache_dir=tempdir
            )
            paddle_model.eval()
            for na, pa in paddle_model.named_parameters():
                print(na)
                print(pa.shape)
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            # 3. forward the torch  model
            import torch
            from transformers import DebertaV2Model

            torch_model = DebertaV2Model.from_pretrained(
                "hf-internal-testing/tiny-random-DebertaV2Model", cache_dir=tempdir
            )
            torch_model.eval()
            for na, pa in torch_model.named_parameters():
                print(na)
                print(pa.shape)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-4,
                )
            )

    @require_package("transformers", "torch")
    def test_deberta_converter_from_local_dir_with_enable_torch(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 2. forward the torch  model
            from transformers import DebertaV2Model

            torch_model = DebertaV2Model.from_pretrained("hf-internal-testing/tiny-random-DebertaV2Model")
            torch_model.save_pretrained(tempdir)

            # 2. forward the paddle model
            from paddlenlp.transformers import model_utils
            from paddlenlp.transformers.deberta_v2.modeling import DebertaV2Model

            model_utils.ENABLE_TORCH_CHECKPOINT = False

            with self.assertRaises(ValueError) as error:
                DebertaV2Model.from_pretrained(tempdir)
                self.assertIn("conversion is been disabled" in str(error.exception))
            model_utils.ENABLE_TORCH_CHECKPOINT = True

    @require_package("transformers", "torch")
    def test_deberta_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import DebertaV2Model

            torch_model = DebertaV2Model.from_pretrained("hf-internal-testing/tiny-random-DebertaV2Model")
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]
            for na, pa in torch_model.named_parameters():
                print(na)
                print(pa.shape)
            print(torch_model.config)

            # 2. forward the paddle model
            from paddlenlp.transformers.deberta_v2.modeling import DebertaV2Model

            paddle_model = DebertaV2Model.from_pretrained(tempdir)
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]
            for na, pa in paddle_model.named_parameters():
                print(na)
                print(pa.shape)
            print(paddle_model.config)

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-4,
                )
            )


if __name__ == "__main__":
    unittest.main()
