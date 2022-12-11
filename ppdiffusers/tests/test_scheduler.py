# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
import json
import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F

import ppdiffusers
from ppdiffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    ScoreSdeVeScheduler,
    VQDiffusionScheduler,
    logging,
)
from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.schedulers.scheduling_utils import SchedulerMixin
from ppdiffusers.utils import deprecate
from ppdiffusers.utils.testing_utils import CaptureLogger


class SchedulerObject(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        e=[1, 3],
    ):
        pass


class SchedulerObject2(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        f=[1, 3],
    ):
        pass


class SchedulerObject3(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        e=[1, 3],
        f=[1, 3],
    ):
        pass


class SchedulerBaseTests(unittest.TestCase):
    def test_save_load_from_different_config(self):
        obj = SchedulerObject()

        # mock add obj class to `ppdiffusers`
        setattr(ppdiffusers, "SchedulerObject", SchedulerObject)
        logger = logging.get_logger("ppdiffusers.configuration_utils")

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)
            with CaptureLogger(logger) as cap_logger_1:
                config = SchedulerObject2.load_config(tmpdirname)
                new_obj_1 = SchedulerObject2.from_config(config)

            # now save a config parameter that is not expected
            with open(os.path.join(tmpdirname, SchedulerObject.config_name), "r") as f:
                data = json.load(f)
                data["unexpected"] = True

            with open(os.path.join(tmpdirname, SchedulerObject.config_name), "w") as f:
                json.dump(data, f)

            with CaptureLogger(logger) as cap_logger_2:
                config = SchedulerObject.load_config(tmpdirname)
                new_obj_2 = SchedulerObject.from_config(config)

            with CaptureLogger(logger) as cap_logger_3:
                config = SchedulerObject2.load_config(tmpdirname)
                new_obj_3 = SchedulerObject2.from_config(config)

        assert new_obj_1.__class__ == SchedulerObject2
        assert new_obj_2.__class__ == SchedulerObject
        assert new_obj_3.__class__ == SchedulerObject2

        assert cap_logger_1.out == ""
        assert (
            cap_logger_2.out
            == "The config attributes {'unexpected': True} were passed to SchedulerObject, but are not expected and"
            " will"
            " be ignored. Please verify your config.json configuration file.\n"
        )
        assert cap_logger_2.out.replace("SchedulerObject", "SchedulerObject2") == cap_logger_3.out

    def test_save_load_compatible_schedulers(self):
        SchedulerObject2._compatibles = ["SchedulerObject"]
        SchedulerObject._compatibles = ["SchedulerObject2"]

        obj = SchedulerObject()

        # mock add obj class to `ppdiffusers`
        setattr(ppdiffusers, "SchedulerObject", SchedulerObject)
        setattr(ppdiffusers, "SchedulerObject2", SchedulerObject2)
        logger = logging.get_logger("ppdiffusers.configuration_utils")

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)

            # now save a config parameter that is expected by another class, but not origin class
            with open(os.path.join(tmpdirname, SchedulerObject.config_name), "r") as f:
                data = json.load(f)
                data["f"] = [0, 0]
                data["unexpected"] = True

            with open(os.path.join(tmpdirname, SchedulerObject.config_name), "w") as f:
                json.dump(data, f)

            with CaptureLogger(logger) as cap_logger:
                config = SchedulerObject.load_config(tmpdirname)
                new_obj = SchedulerObject.from_config(config)

        assert new_obj.__class__ == SchedulerObject

        assert (
            cap_logger.out
            == "The config attributes {'unexpected': True} were passed to SchedulerObject, but are not expected and"
            " will"
            " be ignored. Please verify your config.json configuration file.\n"
        )

    def test_save_load_from_different_config_comp_schedulers(self):
        SchedulerObject3._compatibles = ["SchedulerObject", "SchedulerObject2"]
        SchedulerObject2._compatibles = ["SchedulerObject", "SchedulerObject3"]
        SchedulerObject._compatibles = ["SchedulerObject2", "SchedulerObject3"]

        obj = SchedulerObject()

        # mock add obj class to `ppdiffusers`
        setattr(ppdiffusers, "SchedulerObject", SchedulerObject)
        setattr(ppdiffusers, "SchedulerObject2", SchedulerObject2)
        setattr(ppdiffusers, "SchedulerObject3", SchedulerObject3)
        logger = logging.get_logger("ppdiffusers.configuration_utils")
        logger.setLevel(ppdiffusers.logging.INFO)

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)

            with CaptureLogger(logger) as cap_logger_1:
                config = SchedulerObject.load_config(tmpdirname)
                new_obj_1 = SchedulerObject.from_config(config)

            with CaptureLogger(logger) as cap_logger_2:
                config = SchedulerObject2.load_config(tmpdirname)
                new_obj_2 = SchedulerObject2.from_config(config)

            with CaptureLogger(logger) as cap_logger_3:
                config = SchedulerObject3.load_config(tmpdirname)
                new_obj_3 = SchedulerObject3.from_config(config)

        assert new_obj_1.__class__ == SchedulerObject
        assert new_obj_2.__class__ == SchedulerObject2
        assert new_obj_3.__class__ == SchedulerObject3

        assert cap_logger_1.out == ""
        assert cap_logger_2.out == "{'f'} was not found in config. Values will be initialized to default values.\n"
        assert cap_logger_3.out == "{'f'} was not found in config. Values will be initialized to default values.\n"


class SchedulerCommonTest(unittest.TestCase):
    scheduler_classes = ()
    forward_default_kwargs = ()

    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        sample = paddle.rand((batch_size, num_channels, height, width))

        return sample

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = paddle.arange(num_elems)
        sample = sample.reshape([num_channels, height, width, batch_size])
        sample = sample / num_elems
        sample = sample.transpose([3, 0, 1, 2])

        return sample

    def get_scheduler_config(self):
        raise NotImplementedError

    def dummy_model(self):
        def model(sample, t, *args):
            return sample * t / (t + 1)

        return model

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            # TODO(Suraj) - delete the following two lines once DDPM, DDIM, and PNDM have timesteps casted to float by default
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                time_step = float(time_step)

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, time_step)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Set the seed before step() as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = paddle.Generator().manual_seed(0)
            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = paddle.Generator().manual_seed(0)
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                time_step = float(time_step)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, time_step)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = paddle.Generator().manual_seed(0)
            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = paddle.Generator().manual_seed(0)
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            timestep = 1
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep = float(timestep)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = paddle.Generator().manual_seed(0)
            output = scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = paddle.Generator().manual_seed(0)
            new_output = new_scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_compatibles(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()

            scheduler = scheduler_class(**scheduler_config)

            assert all(c is not None for c in scheduler.compatibles)

            for comp_scheduler_cls in scheduler.compatibles:
                comp_scheduler = comp_scheduler_cls.from_config(scheduler.config)
                assert comp_scheduler is not None

            new_scheduler = scheduler_class.from_config(comp_scheduler.config)

            new_scheduler_config = {k: v for k, v in new_scheduler.config.items() if k in scheduler.config}
            scheduler_diff = {k: v for k, v in new_scheduler.config.items() if k not in scheduler.config}

            # make sure that configs are essentially identical
            assert new_scheduler_config == dict(scheduler.config)

            # make sure that only differences are for configs that are not in init
            init_keys = inspect.signature(scheduler_class.__init__).parameters.keys()
            assert set(scheduler_diff.keys()).intersection(set(init_keys)) == set()

    def test_from_pretrained(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()

            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_pretrained(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            assert scheduler.config == new_scheduler.config

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        timestep_0 = 0
        timestep_1 = 1

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep_0 = float(timestep_0)
                timestep_1 = float(timestep_1)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep_0)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(residual, timestep_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, timestep_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_scheduler_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
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
                        set_nan_tensor_to_zero(tuple_object).cast("float32"),
                        set_nan_tensor_to_zero(dict_object).cast("float32"),
                        atol=1e-5,
                    ),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {paddle.max(paddle.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {paddle.isnan(tuple_object).any()} and `inf`: {paddle.isinf(tuple_object)}. Dict has"
                        f" `nan`: {paddle.isnan(dict_object).any()} and `inf`: {paddle.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", 50)

        timestep = 0
        if len(self.scheduler_classes) > 0 and self.scheduler_classes[0] == IPNDMScheduler:
            timestep = 1

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep = float(timestep)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = paddle.Generator().manual_seed(0)
            outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = paddle.Generator().manual_seed(0)
            outputs_tuple = scheduler.step(residual, timestep, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple, outputs_dict)

    def test_scheduler_public_api(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class != VQDiffusionScheduler:
                self.assertTrue(
                    hasattr(scheduler, "init_noise_sigma"),
                    f"{scheduler_class} does not implement a required attribute `init_noise_sigma`",
                )
                self.assertTrue(
                    hasattr(scheduler, "scale_model_input"),
                    f"{scheduler_class} does not implement a required class method `scale_model_input(sample,"
                    " timestep)`",
                )
            self.assertTrue(
                hasattr(scheduler, "step"),
                f"{scheduler_class} does not implement a required class method `step(...)`",
            )

            if scheduler_class != VQDiffusionScheduler:
                sample = self.dummy_sample
                scaled_sample = scheduler.scale_model_input(sample, 0.0)
                self.assertEqual(sample.shape, scaled_sample.shape)

    def test_add_noise_device(self):
        for scheduler_class in self.scheduler_classes:
            if scheduler_class == IPNDMScheduler:
                continue
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(100)

            sample = self.dummy_sample
            scaled_sample = scheduler.scale_model_input(sample, 0.0)
            self.assertEqual(sample.shape, scaled_sample.shape)

            noise = paddle.randn(scaled_sample.shape, dtype=scaled_sample.dtype)
            t = scheduler.timesteps[5][None]
            noised = scheduler.add_noise(scaled_sample, noise, t)
            self.assertEqual(noised.shape, scaled_sample.shape)

    def test_deprecated_kwargs(self):
        for scheduler_class in self.scheduler_classes:
            has_kwarg_in_model_class = "kwargs" in inspect.signature(scheduler_class.__init__).parameters
            has_deprecated_kwarg = len(scheduler_class._deprecated_kwargs) > 0

            if has_kwarg_in_model_class and not has_deprecated_kwarg:
                raise ValueError(
                    f"{scheduler_class} has `**kwargs` in its __init__ method but has not defined any deprecated"
                    " kwargs under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if"
                    " there are no deprecated arguments or add the deprecated argument with `_deprecated_kwargs ="
                    " [<deprecated_argument>]`"
                )

            if not has_kwarg_in_model_class and has_deprecated_kwarg:
                raise ValueError(
                    f"{scheduler_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated"
                    " kwargs under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs`"
                    f" argument to {self.model_class}.__init__ if there are deprecated arguments or remove the"
                    " deprecated argument from `_deprecated_kwargs = [<deprecated_argument>]`"
                )

    def test_trained_betas(self):
        for scheduler_class in self.scheduler_classes:
            if scheduler_class == VQDiffusionScheduler:
                continue

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config, trained_betas=np.array([0.0, 0.1]))

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_pretrained(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            assert scheduler.betas.tolist() == new_scheduler.betas.tolist()


class DDPMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDPMScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "variance_type": "fixed_small",
            "clip_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [1, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_variance_type(self):
        for variance in ["fixed_small", "fixed_large", "other"]:
            self.check_over_configs(variance_type=variance)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_deprecated_predict_epsilon(self):
        deprecate("remove this test", "0.10.0", "remove")
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()

        sample = self.dummy_sample_deter
        residual = 0.1 * self.dummy_sample_deter
        time_step = 4

        scheduler = scheduler_class(**scheduler_config)
        scheduler_eps = scheduler_class(predict_epsilon=False, **scheduler_config)

        kwargs = {}
        if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
            kwargs["generator"] = paddle.Generator().manual_seed(0)
        output = scheduler.step(residual, time_step, sample, predict_epsilon=False, **kwargs).prev_sample

        kwargs = {}
        if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
            kwargs["generator"] = paddle.Generator().manual_seed(0)
        output_eps = scheduler_eps.step(residual, time_step, sample, predict_epsilon=False, **kwargs).prev_sample

        assert (output - output_eps).abs().sum() < 1e-5

    def test_time_indices(self):
        for t in [0, 500, 999]:
            self.check_over_forward(time_step=t)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert paddle.sum(paddle.abs(scheduler._get_variance(0) - 0.0)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(487) - 0.00979)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(999) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = paddle.Generator().manual_seed(0)

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            pred_prev_sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

            # if t > 0:
            #     noise = self.dummy_sample_deter
            #     variance = scheduler.get_variance(t) ** (0.5) * noise
            #
            # sample = pred_prev_sample + variance
            sample = pred_prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 260.95678711) < 1e-2
        assert abs(result_mean.item() - 0.33978748) < 1e-3

    # TODO check
    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = paddle.Generator().manual_seed(0)

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            pred_prev_sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

            # if t > 0:
            #     noise = self.dummy_sample_deter
            #     variance = scheduler.get_variance(t) ** (0.5) * noise
            #
            # sample = pred_prev_sample + variance
            sample = pred_prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 202.75042724609375) < 1e-2
        assert abs(result_mean.item() - 0.26399797201156616) < 1e-3


class DDIMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDIMScheduler,)
    forward_default_kwargs = (("eta", 0.0), ("num_inference_steps", 50))

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "clip_sample": True,
        }

        config.update(**kwargs)
        return config

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps, eta = 10, 0.0

        model = self.dummy_model()
        sample = self.dummy_sample_deter

        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, eta).prev_sample

        return sample

    def test_timesteps(self):
        for timesteps in [100, 500, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        for steps_offset in [0, 1]:
            self.check_over_configs(steps_offset=steps_offset)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(steps_offset=1)
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(5)
        assert paddle.equal_all(scheduler.timesteps, paddle.to_tensor([801, 601, 401, 201, 1]))

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_time_indices(self):
        for t in [1, 10, 49]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 10, 50], [10, 50, 500]):
            self.check_over_forward(time_step=t, num_inference_steps=num_inference_steps)

    def test_eta(self):
        for t, eta in zip([1, 10, 49], [0.0, 0.5, 1.0]):
            self.check_over_forward(time_step=t, eta=eta)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert paddle.sum(paddle.abs(scheduler._get_variance(0, 0) - 0.0)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(420, 400) - 0.14771)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(980, 960) - 0.32460)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(0, 0) - 0.0)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(487, 486) - 0.00979)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(999, 998) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        sample = self.full_loop()

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 172.0067) < 1e-2
        assert abs(result_mean.item() - 0.223967) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 52.5302) < 1e-2
        assert abs(result_mean.item() - 0.0684) < 1e-3

    def test_full_loop_with_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 149.8295) < 1e-2
        assert abs(result_mean.item() - 0.1951) < 1e-3

    def test_full_loop_with_no_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 149.0784) < 1e-2
        assert abs(result_mean.item() - 0.1941) < 1e-3


class DPMSolverMultistepSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DPMSolverMultistepScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "solver_order": 2,
            "prediction_type": "epsilon",
            "thresholding": False,
            "sample_max_value": 1.0,
            "algorithm_type": "dpmsolver++",
            "solver_type": "midpoint",
            "lower_order_final": False,
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.10]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)
            # copy over dummy past residuals
            scheduler.model_outputs = dummy_past_residuals[: scheduler.config.solver_order]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)
                # copy over dummy past residuals
                new_scheduler.model_outputs = dummy_past_residuals[: new_scheduler.config.solver_order]

            output, new_output = sample, sample
            for t in range(time_step, time_step + scheduler.config.solver_order + 1):
                output = scheduler.step(residual, t, output, **kwargs).prev_sample
                new_output = new_scheduler.step(residual, t, new_output, **kwargs).prev_sample

                assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        pass

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.10]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.model_outputs = dummy_past_residuals[: scheduler.config.solver_order]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)

                # copy over dummy past residual (must be after setting timesteps)
                new_scheduler.model_outputs = dummy_past_residuals[: new_scheduler.config.solver_order]

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        return sample

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # copy over dummy past residuals (must be done after set_timesteps)
            dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.10]
            scheduler.model_outputs = dummy_past_residuals[: scheduler.config.solver_order]

            time_step_0 = scheduler.timesteps[5]
            time_step_1 = scheduler.timesteps[6]

            output_0 = scheduler.step(residual, time_step_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, time_step_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [25, 50, 100, 999, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_thresholding(self):
        self.check_over_configs(thresholding=False)
        for order in [1, 2, 3]:
            for solver_type in ["midpoint", "heun"]:
                for threshold in [0.5, 1.0, 2.0]:
                    for prediction_type in ["epsilon", "sample"]:
                        self.check_over_configs(
                            thresholding=True,
                            prediction_type=prediction_type,
                            sample_max_value=threshold,
                            algorithm_type="dpmsolver++",
                            solver_order=order,
                            solver_type=solver_type,
                        )

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_solver_order_and_type(self):
        for algorithm_type in ["dpmsolver", "dpmsolver++"]:
            for solver_type in ["midpoint", "heun"]:
                for order in [1, 2, 3]:
                    for prediction_type in ["epsilon", "sample"]:
                        self.check_over_configs(
                            solver_order=order,
                            solver_type=solver_type,
                            prediction_type=prediction_type,
                            algorithm_type=algorithm_type,
                        )
                        sample = self.full_loop(
                            solver_order=order,
                            solver_type=solver_type,
                            prediction_type=prediction_type,
                            algorithm_type=algorithm_type,
                        )
                        assert not paddle.isnan(sample).any(), "Samples have nan numbers"

    def test_lower_order_final(self):
        self.check_over_configs(lower_order_final=True)
        self.check_over_configs(lower_order_final=False)

    def test_inference_steps(self):
        for num_inference_steps in [1, 2, 3, 5, 10, 50, 100, 999, 1000]:
            self.check_over_forward(num_inference_steps=num_inference_steps, time_step=0)

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_mean.item() - 0.3301) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_mean.item() - 0.2251) < 1e-3

    # def test_fp16_support(self):
    #     scheduler_class = self.scheduler_classes[0]
    #     scheduler_config = self.get_scheduler_config(thresholding=True, dynamic_thresholding_ratio=0)
    #     scheduler = scheduler_class(**scheduler_config)

    #     num_inference_steps = 10
    #     model = self.dummy_model()
    #     sample = self.dummy_sample_deter.cast("float16")
    #     scheduler.set_timesteps(num_inference_steps)

    #     for i, t in enumerate(scheduler.timesteps):
    #         residual = model(sample, t)
    #         sample = scheduler.step(residual, t, sample).prev_sample

    #     assert sample.dtype == paddle.float16


class PNDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (PNDMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)
            # copy over dummy past residuals
            scheduler.ets = dummy_past_residuals[:]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)
                # copy over dummy past residuals
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step_prk(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step_prk(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step_plms(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step_plms(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        pass

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.ets = dummy_past_residuals[:]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)

                # copy over dummy past residual (must be after setting timesteps)
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step_prk(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step_prk(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step_plms(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step_plms(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(scheduler.prk_timesteps):
            residual = model(sample, t)
            sample = scheduler.step_prk(residual, t, sample).prev_sample

        for i, t in enumerate(scheduler.plms_timesteps):
            residual = model(sample, t)
            sample = scheduler.step_plms(residual, t, sample).prev_sample

        return sample

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # copy over dummy past residuals (must be done after set_timesteps)
            dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]
            scheduler.ets = dummy_past_residuals[:]

            output_0 = scheduler.step_prk(residual, 0, sample, **kwargs).prev_sample
            output_1 = scheduler.step_prk(residual, 1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

            output_0 = scheduler.step_plms(residual, 0, sample, **kwargs).prev_sample
            output_1 = scheduler.step_plms(residual, 1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        for steps_offset in [0, 1]:
            self.check_over_configs(steps_offset=steps_offset)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(steps_offset=1)
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(10)
        assert paddle.equal_all(
            scheduler.timesteps,
            paddle.to_tensor(
                [901, 851, 851, 801, 801, 751, 751, 701, 701, 651, 651, 601, 601, 501, 401, 301, 201, 101, 1]
            ),
        )

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001], [0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_time_indices(self):
        for t in [1, 5, 10]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 5, 10], [10, 50, 100]):
            self.check_over_forward(num_inference_steps=num_inference_steps)

    def test_pow_of_3_inference_steps(self):
        # earlier version of set_timesteps() caused an error indexing alpha's with inference steps as power of 3
        num_inference_steps = 27

        for scheduler_class in self.scheduler_classes:
            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            scheduler.set_timesteps(num_inference_steps)

            # before power of 3 fix, would error on first step, so we only need to do two
            for i, t in enumerate(scheduler.prk_timesteps[:2]):
                sample = scheduler.step_prk(residual, t, sample).prev_sample

    def test_inference_plms_no_past_residuals(self):
        with self.assertRaises(ValueError):
            scheduler_class = self.scheduler_classes[0]
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            scheduler.step_plms(self.dummy_sample, 1, self.dummy_sample).prev_sample

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 198.1318) < 1e-2
        assert abs(result_mean.item() - 0.2580) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 67.3986) < 1e-2
        assert abs(result_mean.item() - 0.0878) < 1e-3

    def test_full_loop_with_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 230.0399) < 1e-2
        assert abs(result_mean.item() - 0.2995) < 1e-3

    def test_full_loop_with_no_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 186.9482) < 1e-2
        assert abs(result_mean.item() - 0.2434) < 1e-3


class ScoreSdeVeSchedulerTest(unittest.TestCase):
    # TODO adapt with class SchedulerCommonTest (scheduler needs Numpy Integration)
    scheduler_classes = (ScoreSdeVeScheduler,)
    forward_default_kwargs = ()

    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        sample = paddle.rand((batch_size, num_channels, height, width))

        return sample

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = paddle.arange(num_elems)
        sample = sample.reshape([num_channels, height, width, batch_size])
        sample = sample / num_elems
        sample = sample.transpose([3, 0, 1, 2])

        return sample

    def dummy_model(self):
        def model(sample, t, *args):
            return sample * t / (t + 1)

        return model

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 2000,
            "snr": 0.15,
            "sigma_min": 0.01,
            "sigma_max": 1348,
            "sampling_eps": 1e-5,
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        for scheduler_class in self.scheduler_classes:
            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            output = scheduler.step_pred(
                residual, time_step, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample
            new_output = new_scheduler.step_pred(
                residual, time_step, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step_correct(
                residual, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample
            new_output = new_scheduler.step_correct(
                residual, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler correction are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        for scheduler_class in self.scheduler_classes:
            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            output = scheduler.step_pred(
                residual, time_step, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample
            new_output = new_scheduler.step_pred(
                residual, time_step, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step_correct(
                residual, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample
            new_output = new_scheduler.step_correct(
                residual, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler correction are not identical"

    def test_timesteps(self):
        for timesteps in [10, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_sigmas(self):
        for sigma_min, sigma_max in zip([0.0001, 0.001, 0.01], [1, 100, 1000]):
            self.check_over_configs(sigma_min=sigma_min, sigma_max=sigma_max)

    def test_time_indices(self):
        for t in [0.1, 0.5, 0.75]:
            self.check_over_forward(time_step=t)

    def test_full_loop_no_noise(self):
        kwargs = dict(self.forward_default_kwargs)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 3

        model = self.dummy_model()
        sample = self.dummy_sample_deter

        scheduler.set_sigmas(num_inference_steps)
        scheduler.set_timesteps(num_inference_steps)
        generator = paddle.Generator().manual_seed(0)

        for i, t in enumerate(scheduler.timesteps):
            sigma_t = scheduler.sigmas[i]

            for _ in range(scheduler.config.correct_steps):
                with paddle.no_grad():
                    model_output = model(sample, sigma_t)
                sample = scheduler.step_correct(model_output, sample, generator=generator, **kwargs).prev_sample

            with paddle.no_grad():
                model_output = model(sample, sigma_t)

            output = scheduler.step_pred(model_output, t, sample, generator=generator, **kwargs)
            sample, _ = output.prev_sample, output.prev_sample_mean

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert np.isclose(result_sum.item(), 13210036224.0)
        assert np.isclose(result_mean.item(), 17200568.0)

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step_pred(
                residual, 0, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample
            output_1 = scheduler.step_pred(
                residual, 1, sample, generator=paddle.Generator().manual_seed(0), **kwargs
            ).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)


class LMSDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (LMSDiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_time_indices(self):
        for t in [0, 500, 800]:
            self.check_over_forward(time_step=t)

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 1006.388) < 1e-2
        assert abs(result_mean.item() - 1.31) < 1e-3

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 0.0017) < 1e-2
        assert abs(result_mean.item() - 2.2676e-06) < 1e-3


class EulerDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EulerDiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        generator = paddle.Generator().manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 10.0807) < 1e-2
        assert abs(result_mean.item() - 0.0131) < 1e-3

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        generator = paddle.Generator().manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 0.0002) < 1e-2
        assert abs(result_mean.item() - 2.2676e-06) < 1e-3


class EulerAncestralDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EulerAncestralDiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        generator = paddle.Generator().manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 144.8084) < 1e-2
        assert abs(result_mean.item() - 0.18855) < 1e-3

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        generator = paddle.Generator().manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 102.5807) < 1e-2
        assert abs(result_mean.item() - 0.1335) < 1e-3


class IPNDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (IPNDMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

    def get_scheduler_config(self, **kwargs):
        config = {"num_train_timesteps": 1000}
        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)
            # copy over dummy past residuals
            scheduler.ets = dummy_past_residuals[:]

            if time_step is None:
                time_step = scheduler.timesteps[len(scheduler.timesteps) // 2]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)
                # copy over dummy past residuals
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        pass

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.ets = dummy_past_residuals[:]

            if time_step is None:
                time_step = scheduler.timesteps[len(scheduler.timesteps) // 2]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)

                # copy over dummy past residual (must be after setting timesteps)
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert paddle.sum(paddle.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        return sample

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # copy over dummy past residuals (must be done after set_timesteps)
            dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]
            scheduler.ets = dummy_past_residuals[:]

            time_step_0 = scheduler.timesteps[5]
            time_step_1 = scheduler.timesteps[6]

            output_0 = scheduler.step(residual, time_step_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, time_step_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

            output_0 = scheduler.step(residual, time_step_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, time_step_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps, time_step=None)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 5, 10], [10, 50, 100]):
            self.check_over_forward(num_inference_steps=num_inference_steps, time_step=None)

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_mean.item() - 2540529) < 10


class VQDiffusionSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (VQDiffusionScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_vec_classes": 4097,
            "num_train_timesteps": 100,
        }

        config.update(**kwargs)
        return config

    def dummy_sample(self, num_vec_classes):
        batch_size = 4
        height = 8
        width = 8

        sample = paddle.randint(0, num_vec_classes, (batch_size, height * width))

        return sample

    @property
    def dummy_sample_deter(self):
        assert False

    def dummy_model(self, num_vec_classes):
        def model(sample, t, *args):
            batch_size, num_latent_pixels = sample.shape
            logits = paddle.rand((batch_size, num_vec_classes - 1, num_latent_pixels))
            return_value = F.log_softmax(logits.cast("float64"), axis=1).cast("float32")
            return return_value

        return model

    def test_timesteps(self):
        for timesteps in [2, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_num_vec_classes(self):
        for num_vec_classes in [5, 100, 1000, 4000]:
            self.check_over_configs(num_vec_classes=num_vec_classes)

    def test_time_indices(self):
        for t in [0, 50, 99]:
            self.check_over_forward(time_step=t)

    def test_add_noise_device(self):
        pass


class HeunDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (HeunDiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 0.0010020951740443707) < 1e-2
        assert abs(result_mean.item() - 1.3048115761193912e-06) < 1e-3

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 4.693428650170972e-07) < 1e-2
        assert abs(result_mean.item() - 0.0002) < 1e-3


class KDPM2DiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KDPM2DiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 4.693428650170972e-07) < 1e-2
        assert abs(result_mean.item() - 0.0002) < 1e-3

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 20.4125) < 1e-2
        assert abs(result_mean.item() - 0.0266) < 1e-3


class KDPM2AncestralDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KDPM2AncestralDiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        generator = paddle.Generator().manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 13913.0625) < 1e-2
        assert abs(result_mean.item() - 18.115968704223633) < 5e-3

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        generator = paddle.Generator().manual_seed(0)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 327.8027) < 1e-2
        assert abs(result_mean.item() - 0.4268) < 1e-3
