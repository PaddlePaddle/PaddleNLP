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
from tempfile import TemporaryDirectory

import numpy as np
import paddle
from PIL import Image

from paddlenlp.taskflow import Taskflow
from paddlenlp.taskflow.multimodal_feature_extraction import (
    MultimodalFeatureExtractionTask,
)


class TestMultimodalFeatureExtractionTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TemporaryDirectory()
        cls.batch_size = 2
        cls.max_resolution = 40
        cls.min_resolution = 30
        cls.num_channels = 3
        cls.max_length = 30

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_model_np(self):
        feature_extractor = Taskflow(
            model="__internal_testing__/tiny-random-ernievil2",
            task="feature_extraction",
            return_tensors="np",
            max_length=self.max_length,
        )
        outputs = feature_extractor("This is a test")
        self.assertEqual(outputs["features"].shape, (1, 32))

    def test_return_tensors(self):
        feature_extractor = Taskflow(
            model="__internal_testing__/tiny-random-ernievil2",
            task="feature_extraction",
            return_tensors="pd",
            max_length=self.max_length,
        )
        outputs = feature_extractor(
            "This is a test",
        )
        self.assertTrue(paddle.is_tensor(outputs["features"]))

    def prepare_inputs(self, equal_resolution=False, numpify=False, paddleify=False):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PaddlePaddle tensors if one specifies paddleify=True.
        """

        assert not (numpify and paddleify), "You cannot specify both numpy and PaddlePaddle tensors at the same time"

        if equal_resolution:
            image_inputs = []
            for i in range(self.batch_size):
                image_inputs.append(
                    np.random.randint(
                        255, size=(self.num_channels, self.max_resolution, self.max_resolution), dtype=np.uint8
                    )
                )
        else:
            image_inputs = []
            for i in range(self.batch_size):
                width, height = np.random.choice(np.arange(self.min_resolution, self.max_resolution), 2)
                image_inputs.append(np.random.randint(255, size=(self.num_channels, width, height), dtype=np.uint8))

        if not numpify and not paddleify:
            # PIL expects the channel dimension as last dimension
            image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        if paddleify:
            image_inputs = [paddle.to_tensor(x) for x in image_inputs]

        return image_inputs

    def test_feature_extraction_task(self):
        input_text = (["这是一只猫", "这是一只狗"],)
        # dygraph text test
        dygraph_taskflow = MultimodalFeatureExtractionTask(
            model="__internal_testing__/tiny-random-ernievil2",
            task="feature_extraction",
            is_static_model=False,
            return_tensors="np",
            max_length=self.max_length,
        )
        dygraph_results = dygraph_taskflow(input_text)
        shape = dygraph_results["features"].shape
        self.assertEqual(shape[0], 2)
        # static text test
        static_taskflow = MultimodalFeatureExtractionTask(
            model="__internal_testing__/tiny-random-ernievil2",
            task="feature_extraction",
            is_static_model=True,
            return_tensors="np",
            device_id=0,
            max_length=self.max_length,
        )
        static_results = static_taskflow(input_text)
        self.assertEqual(static_results["features"].shape[0], 2)

        for dygraph_result, static_result in zip(dygraph_results["features"], static_results["features"]):
            for dygraph_pred, static_pred in zip(dygraph_result.tolist(), static_result.tolist()):
                self.assertAlmostEqual(dygraph_pred, static_pred, delta=1e-5)

        input_image = (self.prepare_inputs(equal_resolution=True, paddleify=False),)
        #  dygraph image test
        dygraph_results = dygraph_taskflow(input_image)
        self.assertEqual(dygraph_results["features"].shape[0], 2)

        # static image test
        static_results = static_taskflow(input_image)
        self.assertEqual(static_results["features"].shape[0], 2)

        for dygraph_result, static_result in zip(dygraph_results["features"], static_results["features"]):
            for dygraph_pred, static_pred in zip(dygraph_result.tolist(), static_result.tolist()):
                self.assertAlmostEqual(dygraph_pred, static_pred, delta=1e-5)

    @unittest.skip("numerical error")
    def test_taskflow_task(self):
        input_text = ["这是一只猫", "这是一只狗"]

        # dygraph test
        dygraph_taskflow = Taskflow(
            model="__internal_testing__/tiny-random-ernievil2",
            task="feature_extraction",
            is_static_model=False,
            return_tensors="np",
            max_length=self.max_length,
        )
        dygraph_results = dygraph_taskflow(input_text)
        shape = dygraph_results["features"].shape

        self.assertEqual(shape[0], 2)
        # static test
        static_taskflow = Taskflow(
            model="__internal_testing__/tiny-random-ernievil2",
            task="feature_extraction",
            is_static_model=True,
            return_tensors="np",
            max_length=self.max_length,
        )
        static_results = static_taskflow(input_text)
        self.assertEqual(static_results["features"].shape[0], 2)
        for dygraph_result, static_result in zip(dygraph_results["features"], static_results["features"]):
            for dygraph_pred, static_pred in zip(dygraph_result.tolist(), static_result.tolist()):
                self.assertAlmostEqual(dygraph_pred, static_pred, delta=1e-5)

        input_image = self.prepare_inputs(equal_resolution=True, paddleify=False)
        #  dygraph image test
        dygraph_results = dygraph_taskflow(input_image)
        self.assertEqual(dygraph_results["features"].shape[0], 2)

        # static image test
        static_results = static_taskflow(input_image)
        self.assertEqual(static_results["features"].shape[0], 2)

        for dygraph_result, static_result in zip(dygraph_results["features"], static_results["features"]):
            for dygraph_pred, static_pred in zip(dygraph_result.tolist(), static_result.tolist()):
                self.assertAlmostEqual(dygraph_pred, static_pred, delta=1e-5)
