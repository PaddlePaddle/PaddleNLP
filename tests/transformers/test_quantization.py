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

# ****** Unitest not yet support paddlepaddle>=2.5.2 ******
# import unittest
# from tempfile import TemporaryDirectory

# import numpy as np
# import paddle
# from paddle.nn.quant import weight_quantize
# from parameterized import parameterized

# from paddlenlp.transformers import AutoModel
# from paddlenlp.utils.quantization import QuantizationLinear


# class TestQuantizationLinear(unittest.TestCase):
#     @parameterized.expand([("weight_only_int8",), ("weight_only_int4",), ("llm.int8",)])
#     def test_forward(self, quant_algo):
#         qlinear = QuantizationLinear(in_features=64, out_features=64, quant_algo=quant_algo, dtype="float16")
#         x = paddle.randn([2, 4, 64], "float16")
#         weight = paddle.randn([64, 64], "float16")
#         quant_weight, quant_scale = weight_quantize(weight, quant_algo)
#         qlinear.quant_weight.set_value(quant_weight)
#         qlinear.quant_scale.set_value(quant_scale)

#         output = qlinear(x)
#         self.assertEqual(output.shape, [2, 4, 64])


# class TestQuantizationModel(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.model1 = AutoModel.from_pretrained(
#             "__internal_testing__/test-tiny-random-llama",
#             quantization_config={"quant_algo": "weight_only_int8"},
#             dtype="float16",
#         )
#         cls.model2 = AutoModel.from_pretrained(
#             "__internal_testing__/test-tiny-random-llama",
#             quantization_config={"quant_algo": "weight_only_int4"},
#             dtype="float16",
#         )
#         cls.model3 = AutoModel.from_pretrained(
#             "__internal_testing__/test-tiny-random-llama",
#             quantization_config={"quant_algo": "llm.int8"},
#             dtype="float16",
#         )
#         cls.model1.eval()
#         cls.model2.eval()
#         cls.model3.eval()

#     def test_forward(self):
#         input_ids = paddle.to_tensor(np.random.randint(0, 128, [1, 20]))
#         for model in [self.model1, self.model2, self.model3]:
#             output = model(input_ids)
#             self.assertEqual(output[0].shape, [1, 20, 64])

#     def test_save_pretrained(self):
#         input_ids = paddle.to_tensor(np.random.randint(0, 128, [1, 20]))
#         for model in [self.model1, self.model2, self.model3]:
#             with TemporaryDirectory() as tempdir:
#                 model.save_pretrained(tempdir)
#                 model_save = AutoModel.from_pretrained(tempdir)
#                 self.assertTrue(paddle.allclose(model(input_ids)[0], model_save(input_ids)[0]))
