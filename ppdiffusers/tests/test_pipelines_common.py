# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import contextlib
import gc
import inspect
import io
import re
import tempfile
import time
from typing import Callable, Union

import numpy as np
import paddle

import ppdiffusers
from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils import logging
from ppdiffusers.utils.testing_utils import require_paddle

ALLOWED_REQUIRED_ARGS = ["source_prompt", "prompt", "image", "mask_image", "example_image"]


@require_paddle
class PipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each Paddle pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    # set these parameters to False in the child class if the pipeline does not support the corresponding functionality
    test_attention_slicing = True

    def get_generator(self, seed):
        generator = paddle.Generator().manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Union[Callable, DiffusionPipeline]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_components(self):
        raise NotImplementedError(
            "You need to implement `get_dummy_components(self)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, device, seed=0):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self, device, seed)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def tearDown(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_save_load_local(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 1e-4)

    def test_pipeline_call_implements_required_args(self):
        assert hasattr(self.pipeline_class, "__call__"), f"{self.pipeline_class} should have a `__call__` method"
        parameters = inspect.signature(self.pipeline_class.__call__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        required_parameters.pop("self")
        required_parameters = set(required_parameters)
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})

        for param in required_parameters:
            if param == "kwargs":
                # kwargs can be added if arguments of pipeline call function are deprecated
                continue
            assert param in ALLOWED_REQUIRED_ARGS

        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})

        required_optional_params = ["generator", "num_inference_steps", "return_dict"]
        for param in required_optional_params:
            assert param in optional_parameters

    def test_inference_batch_consistent(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=ppdiffusers.logging.FATAL)

        # batchify inputs
        for batch_size in [2, 4, 13]:
            batched_inputs = {}
            for name, value in inputs.items():
                if name in ALLOWED_REQUIRED_ARGS:
                    # prompt is string
                    if name == "prompt":
                        len_prompt = len(value)
                        # make unequal batch sizes
                        batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                        # make last batch super long
                        batched_inputs[name][-1] = 2000 * "very long"
                    # or else we have images
                    else:
                        batched_inputs[name] = batch_size * [value]
                elif name == "batch_size":
                    batched_inputs[name] = batch_size
                else:
                    batched_inputs[name] = value

            batched_inputs["num_inference_steps"] = inputs["num_inference_steps"]
            batched_inputs["output_type"] = None

            if self.pipeline_class.__name__ == "DanceDiffusionPipeline":
                batched_inputs.pop("output_type")

            output = pipe(**batched_inputs)

            assert len(output[0]) == batch_size

            batched_inputs["output_type"] = "np"

            if self.pipeline_class.__name__ == "DanceDiffusionPipeline":
                batched_inputs.pop("output_type")

            output = pipe(**batched_inputs)[0]

            assert output.shape[0] == batch_size

        logger.setLevel(level=ppdiffusers.logging.WARNING)

    def test_inference_batch_single_identical(self):
        if self.pipeline_class.__name__ in ["CycleDiffusionPipeline", "RePaintPipeline"]:
            # RePaint can hardly be made deterministic since the scheduler is currently always
            # indeterministic
            # CycleDiffusion is also slighly undeterministic
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=ppdiffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batch_size = 3
        for name, value in inputs.items():
            if name in ALLOWED_REQUIRED_ARGS:
                # prompt is string
                if name == "prompt":
                    len_prompt = len(value)
                    # make unequal batch sizes
                    batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                    # make last batch super long
                    batched_inputs[name][-1] = 2000 * "very long"
                # or else we have images
                else:
                    batched_inputs[name] = batch_size * [value]
            elif name == "batch_size":
                batched_inputs[name] = batch_size
            elif name == "generator":
                batched_inputs[name] = [self.get_generator(i) for i in range(batch_size)]
            else:
                batched_inputs[name] = value

        batched_inputs["num_inference_steps"] = inputs["num_inference_steps"]

        if self.pipeline_class.__name__ != "DanceDiffusionPipeline":
            batched_inputs["output_type"] = "np"

        output_batch = pipe(**batched_inputs)
        assert output_batch[0].shape[0] == batch_size

        inputs["generator"] = self.get_generator(0)

        output = pipe(**inputs)

        logger.setLevel(level=ppdiffusers.logging.WARNING)
        assert np.abs(output_batch[0][0] - output[0][0]).max() < 1e-4

    def test_dict_tuple_outputs_equivalent(self):

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs())[0]
        output_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        max_diff = np.abs(output - output_tuple).max()
        self.assertLess(max_diff, 1e-4)

    def test_num_inference_steps_consistent(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        outputs = []
        times = []
        for num_steps in [9, 6, 3]:
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = num_steps

            start_time = time.time()
            output = pipe(**inputs)[0]
            inference_time = time.time() - start_time

            outputs.append(output)
            times.append(inference_time)

        # check that all outputs have the same shape
        self.assertTrue(all(outputs[0].shape == output.shape for output in outputs))
        # check that the inference time increases with the number of inference steps
        self.assertTrue(all(times[i] < times[i - 1] for i in range(1, len(times))))

    def test_components_function(self):
        init_components = self.get_dummy_components()
        pipe = self.pipeline_class(**init_components)

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def test_save_load_optional_components(self):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 1e-4)

    def test_attention_slicing_forward_pass(self):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        # Warmup pass when using mps (see #372)
        inputs = self.get_dummy_inputs()
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs()
        output_with_slicing = pipe(**inputs)[0]

        max_diff = np.abs(output_with_slicing - output_without_slicing).max()
        self.assertLess(max_diff, 1e-3, "Attention slicing should not affect the inference results")

    def test_progress_bar(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        inputs = self.get_dummy_inputs()
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")
