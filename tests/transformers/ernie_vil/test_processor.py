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

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest
from PIL import Image

from paddlenlp.transformers import (
    ErnieViLImageProcessor,
    ErnieViLProcessor,
    ErnieViLTokenizer,
)


class ErnieViLProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "的",
            "价",
            "格",
            "是",
            "15",
            "便",
            "alex",
            "##andra",
            "，",
            "。",
            "-",
            "t",
            "shirt",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, ErnieViLTokenizer.resource_files_names["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        image_processor_map = {
            "do_resize": True,
            "size": 224,
            "do_center_crop": True,
            "crop_size": {"height": 18, "width": 18},
            "do_normalize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_convert_rgb": True,
        }
        self.image_processor_file = os.path.join(self.tmpdirname, "preprocessor_config.json")
        with open(self.image_processor_file, "w", encoding="utf-8") as fp:
            json.dump(image_processor_map, fp)

    def get_tokenizer(self, **kwargs):
        return ErnieViLTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return ErnieViLImageProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor_slow = ErnieViLProcessor(tokenizer=tokenizer_slow, image_processor=image_processor)
        processor_slow.save_pretrained(self.tmpdirname)
        processor_slow = ErnieViLProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, ErnieViLTokenizer)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, ErnieViLImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        processor = ErnieViLProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(cls_token="(CLS)", sep_token="(SEP)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False)

        processor = ErnieViLProcessor.from_pretrained(
            self.tmpdirname, cls_token="(CLS)", sep_token="(SEP)", do_normalize=False
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, ErnieViLImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ErnieViLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ErnieViLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "Alexandra，T-shirt的价格是15便士。"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ErnieViLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "Alexandra，T-shirt的价格是15便士。"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["input_ids", "pixel_values"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ErnieViLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ErnieViLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "Alexandra，T-shirt的价格是15便士。"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)
