# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from paddlenlp.transformers import (
    LlamaTokenizer,
    MiniGPT4ImageProcessor,
    MiniGPT4Processor,
)


class MiniGPT4ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = MiniGPT4ImageProcessor.from_pretrained("minigpt4-13b")
        tokenizer = LlamaTokenizer.from_pretrained("minigpt4-13b")
        processor = MiniGPT4Processor(image_processor, tokenizer)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return MiniGPT4Processor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return MiniGPT4Processor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PaddlePaddle tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def test_save_load_pretrained_additional_features(self):
        processor = MiniGPT4Processor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, rescale_factor=1.0)

        processor = MiniGPT4Processor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, rescale_factor=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, MiniGPT4ImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MiniGPT4Processor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor.process_images(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = MiniGPT4Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = ["lower newer"]
        encoded_processor = processor.process_texts(texts=input_str, return_tensors="np")

        first_texts = "###Human: <Img>"
        second_texts = "</Img> lower newer###Assistant: "
        first_text_encoding = tokenizer(
            text=first_texts,
            return_tensors="np",
            add_special_tokens=True,
        )
        second_text_encoding = tokenizer(
            text=second_texts,
            return_tensors="np",
            add_special_tokens=False,
        )

        encoded_tok = {
            "first_input_ids": first_text_encoding["input_ids"],
            "first_attention_mask": first_text_encoding["attention_mask"],
            "second_input_ids": second_text_encoding["input_ids"],
            "second_attention_mask": second_text_encoding["attention_mask"],
        }

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key].tolist(), encoded_processor[key].tolist())

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MiniGPT4Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            ["pixel_values", "first_input_ids", "first_attention_mask", "second_input_ids", "second_attention_mask"],
        )

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MiniGPT4Processor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = MiniGPT4Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input)

        # For now the processor supports only ['pixel_values', 'input_ids', 'attention_mask']
        self.assertListEqual(
            list(inputs.keys()),
            ["pixel_values", "first_input_ids", "first_attention_mask", "second_input_ids", "second_attention_mask"],
        )
