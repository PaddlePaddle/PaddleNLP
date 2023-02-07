# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import paddle
from PIL import Image

from paddlenlp.transformers import ErnieViLImageProcessor

from ..test_image_processing_common import ImageProcessingSavingTestMixin


class ErnieViLImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_center_crop=True,
        crop_size=None,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"shortest_edge": 224}
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

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


class ErnieViLImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = ErnieViLImageProcessor

    def setUp(self):
        self.image_processor_tester = ErnieViLImageProcessingTester(self, do_center_crop=True)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_center_crop"))
        self.assertTrue(hasattr(image_processing, "center_crop"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 224})
        self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
        self.assertEqual(image_processor.size, 42)
        self.assertEqual(image_processor.crop_size, 84)

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pd").pixel_values
        self.assertEqual(
            encoded_images.shape,
            [
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ],
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pd").pixel_values
        self.assertEqual(
            encoded_images.shape,
            [
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ],
        )

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="np").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="np").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )

    def test_call_paddle(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PaddlePaddle tensors
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False, paddleify=True)
        for image in image_inputs:
            self.assertIsInstance(image, paddle.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pd").pixel_values
        self.assertEqual(
            encoded_images.shape,
            [
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ],
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pd").pixel_values
        self.assertEqual(
            encoded_images.shape,
            [
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ],
        )


class ErnieViLImageProcessingTestFourChannels(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = ErnieViLImageProcessor

    def setUp(self):
        self.image_processor_tester = ErnieViLImageProcessingTester(self, num_channels=4, do_center_crop=True)
        self.expected_encoded_image_num_channels = 3

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_center_crop"))
        self.assertTrue(hasattr(image_processing, "center_crop"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    def test_batch_feature(self):
        pass

    def test_call_pil_four_channels(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pd").pixel_values
        self.assertEqual(
            encoded_images.shape,
            [
                1,
                self.expected_encoded_image_num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ],
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pd").pixel_values
        self.assertEqual(
            encoded_images.shape,
            [
                self.image_processor_tester.batch_size,
                self.expected_encoded_image_num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ],
        )
