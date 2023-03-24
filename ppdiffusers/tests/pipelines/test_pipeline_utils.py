# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from ppdiffusers.pipelines.pipeline_utils import is_safetensors_compatible


class IsSafetensorsCompatibleTests(unittest.TestCase):
    def test_all_is_compatible(self):
        filenames = [
            "safety_checker/pytorch_model.bin",
            "safety_checker/model.safetensors",
            "vae/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_model_is_compatible(self):
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_diffusers_model_is_not_compatible(self):
        filenames = [
            "safety_checker/pytorch_model.bin",
            "safety_checker/model.safetensors",
            "vae/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.bin",
            # Removed: 'unet/diffusion_pytorch_model.safetensors',
        ]
        self.assertFalse(is_safetensors_compatible(filenames))

    def test_transformer_model_is_compatible(self):
        filenames = [
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.safetensors",
        ]
        self.assertTrue(is_safetensors_compatible(filenames))

    def test_transformer_model_is_not_compatible(self):
        filenames = [
            "safety_checker/pytorch_model.bin",
            "safety_checker/model.safetensors",
            "vae/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/pytorch_model.bin",
            # Removed: 'text_encoder/model.safetensors',
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        self.assertFalse(is_safetensors_compatible(filenames))

    def test_all_is_compatible_variant(self):
        filenames = [
            "safety_checker/pytorch_model.fp16.bin",
            "safety_checker/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.fp16.bin",
            "text_encoder/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

    def test_diffusers_model_is_compatible_variant(self):
        filenames = [
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

    def test_diffusers_model_is_compatible_variant_partial(self):
        # pass variant but use the non-variant filenames
        filenames = [
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

    def test_diffusers_model_is_not_compatible_variant(self):
        filenames = [
            "safety_checker/pytorch_model.fp16.bin",
            "safety_checker/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.fp16.bin",
            "text_encoder/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.bin",
            # Removed: 'unet/diffusion_pytorch_model.fp16.safetensors',
        ]
        variant = "fp16"
        self.assertFalse(is_safetensors_compatible(filenames, variant=variant))

    def test_transformer_model_is_compatible_variant(self):
        filenames = [
            "text_encoder/pytorch_model.fp16.bin",
            "text_encoder/model.fp16.safetensors",
        ]
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

    def test_transformer_model_is_compatible_variant_partial(self):
        # pass variant but use the non-variant filenames
        filenames = [
            "text_encoder/pytorch_model.bin",
            "text_encoder/model.safetensors",
        ]
        variant = "fp16"
        self.assertTrue(is_safetensors_compatible(filenames, variant=variant))

    def test_transformer_model_is_not_compatible_variant(self):
        filenames = [
            "safety_checker/pytorch_model.fp16.bin",
            "safety_checker/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/pytorch_model.fp16.bin",
            # 'text_encoder/model.fp16.safetensors',
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ]
        variant = "fp16"
        self.assertFalse(is_safetensors_compatible(filenames, variant=variant))
