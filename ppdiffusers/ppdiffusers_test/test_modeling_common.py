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

import inspect
import tempfile
import unittest
import unittest.mock as mock
from typing import Dict, List, Tuple

import numpy as np
import paddle
from requests.exceptions import HTTPError

from ppdiffusers.models import UNet2DConditionModel
from ppdiffusers.training_utils import EMAModel


class ModelUtilsTest(unittest.TestCase):
    def test_cached_files_are_used_when_no_internet(self):
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        orig_model = UNet2DConditionModel.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet"
        )
        with mock.patch("requests.request", return_value=response_mock):
            model = UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet", local_files_only=True
            )
        for p1, p2 in zip(orig_model.parameters(), model.parameters()):
            if (p1 != p2).cast("int64").sum() > 0:
                assert False, "Parameters not the same!"


class ModelTesterMixin:
    def test_from_save_pretrained(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = self.model_class.from_pretrained(tmpdirname)
        with paddle.no_grad():
            image = model(**inputs_dict)
            if isinstance(image, dict):
                image = image.sample
            new_image = new_model(**inputs_dict)
            if isinstance(new_image, dict):
                new_image = new_image.sample
        max_diff = (image - new_image).abs().sum().item()
        self.assertLessEqual(max_diff, 5e-05, "Models give different forward passes")

    def test_from_save_pretrained_variant(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, variant="fp16")
            new_model = self.model_class.from_pretrained(tmpdirname, variant="fp16")
            # non-variant cannot be loaded
            with self.assertRaises(OSError) as error_context:
                self.model_class.from_pretrained(tmpdirname)

            # make sure that error message states what keys are missing
            # support diffusion_pytorch_model.bin and model_state.pdparams
            assert "Error no file named model_state.pdparams found in directory" in str(
                error_context.exception
            ) or "Error no file named diffusion_pytorch_model.bin found in directory" in str(error_context.exception)
        with paddle.no_grad():

            image = model(**inputs_dict)
            if isinstance(image, dict):
                image = image.sample
            new_image = new_model(**inputs_dict)
            if isinstance(new_image, dict):
                new_image = new_image.sample
        max_diff = (image - new_image).abs().sum().item()
        self.assertLessEqual(max_diff, 5e-05, "Models give different forward passes")

    def test_from_save_pretrained_dtype(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()
        for dtype in [paddle.float32, paddle.float16, paddle.bfloat16]:

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.to(dtype=dtype)
                model.save_pretrained(tmpdirname)
                new_model = self.model_class.from_pretrained(tmpdirname, paddle_dtype=dtype)
                assert new_model.dtype == dtype
                new_model = self.model_class.from_pretrained(tmpdirname, paddle_dtype=dtype)
                assert new_model.dtype == dtype

    def test_determinism(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()
        with paddle.no_grad():
            first = model(**inputs_dict)
            if isinstance(first, dict):
                first = first.sample
            second = model(**inputs_dict)
            if isinstance(second, dict):
                second = second.sample
        out_1 = first.cpu().numpy()
        out_2 = second.cpu().numpy()
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, 1e-05)

    def test_output(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()
        with paddle.no_grad():
            output = model(**inputs_dict)
            if isinstance(output, dict):
                output = output.sample
        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_forward_with_norm_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = 16, 32
        model = self.model_class(**init_dict)
        model.eval()
        with paddle.no_grad():
            output = model(**inputs_dict)
            if isinstance(output, dict):
                output = output.sample
        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_forward_signature(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        signature = inspect.signature(model.forward)
        arg_names = [*signature.parameters.keys()]
        expected_arg_names = ["sample", "timestep"]
        self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_model_from_pretrained(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = self.model_class.from_pretrained(tmpdirname)
            new_model.eval()
        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            self.assertEqual(param_1.shape, param_2.shape)
        with paddle.no_grad():
            output_1 = model(**inputs_dict)
            if isinstance(output_1, dict):
                output_1 = output_1.sample
            output_2 = new_model(**inputs_dict)
            if isinstance(output_2, dict):
                output_2 = output_2.sample
        self.assertEqual(output_1.shape, output_2.shape)

    def test_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.train()
        output = model(**inputs_dict)
        if isinstance(output, dict):
            output = output.sample
        noise = paddle.randn(shape=list((inputs_dict["sample"].shape[0],) + self.output_shape))
        loss = paddle.nn.functional.mse_loss(input=output, label=noise)
        loss.backward()

    def test_ema_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.train()
        ema_model = EMAModel(model.parameters())
        output = model(**inputs_dict)
        if isinstance(output, dict):
            output = output.sample
        # breakpoint()
        noise = paddle.randn(shape=list((inputs_dict["sample"].shape[0],) + self.output_shape))
        loss = paddle.nn.functional.mse_loss(input=output, label=noise)
        loss.backward()
        ema_model.step(model.parameters())

    def test_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            # t[t != t] = 0
            return t

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    paddle.allclose(
                        set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-05
                    ),
                    msg=f"Tuple and dict output are not equal. Difference: {paddle.max(x=paddle.abs(x=tuple_object - dict_object))}. Tuple has `nan`: {paddle.isnan(x=tuple_object).any()} and `inf`: {paddle.isinf(x=tuple_object)}. Dict has `nan`: {paddle.isnan(x=dict_object).any()} and `inf`: {paddle.isinf(x=dict_object)}.",
                )

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()
        with paddle.no_grad():
            outputs_dict = model(**inputs_dict)
            outputs_tuple = model(**inputs_dict, return_dict=False)
        recursive_check(outputs_tuple, outputs_dict)

    def test_enable_disable_gradient_checkpointing(self):
        if not self.model_class._supports_gradient_checkpointing:
            return
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        self.assertFalse(model.is_gradient_checkpointing)
        model.enable_gradient_checkpointing()
        self.assertTrue(model.is_gradient_checkpointing)
        model.disable_gradient_checkpointing()
        self.assertFalse(model.is_gradient_checkpointing)

    def test_deprecated_kwargs(self):
        has_kwarg_in_model_class = "kwargs" in inspect.signature(self.model_class.__init__).parameters
        has_deprecated_kwarg = len(self.model_class._deprecated_kwargs) > 0
        if has_kwarg_in_model_class and not has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} has `**kwargs` in its __init__ method but has not defined any deprecated kwargs under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if there are no deprecated arguments or add the deprecated argument with `_deprecated_kwargs = [<deprecated_argument>]`"
            )
        if not has_kwarg_in_model_class and has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated kwargs under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs` argument to {self.model_class}.__init__ if there are deprecated arguments or remove the deprecated argument from `_deprecated_kwargs = [<deprecated_argument>]`"
            )
