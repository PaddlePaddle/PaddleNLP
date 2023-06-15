# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# This file is modified from
#  https://github.com/huggingface/transformers/blob/main/src/transformers/integrations.py

import importlib
import json

from ..peft import LoRAModel, PrefixModelForCausalLM
from ..transformers import PretrainedModel
from ..utils.log import logger
from .trainer_callback import TrainerCallback


def is_visualdl_available():
    return importlib.util.find_spec("visualdl") is not None


def is_ray_available():
    return importlib.util.find_spec("ray.air") is not None


def get_available_reporting_integrations():
    integrations = []
    if is_visualdl_available():
        integrations.append("visualdl")

    return integrations


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class VisualDLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [VisualDL](https://www.paddlepaddle.org.cn/paddle/visualdl).
    Args:
        vdl_writer (`LogWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, vdl_writer=None):
        has_visualdl = is_visualdl_available()
        if not has_visualdl:
            raise RuntimeError("VisualDLCallback requires visualdl to be installed. Please install visualdl.")
        if has_visualdl:
            try:
                from visualdl import LogWriter

                self._LogWriter = LogWriter
            except ImportError:
                self._LogWriter = None
        else:
            self._LogWriter = None
        self.vdl_writer = vdl_writer

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._LogWriter is not None:
            self.vdl_writer = self._LogWriter(logdir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if self.vdl_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.vdl_writer is not None:
            self.vdl_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
                    model = kwargs["model"].model
                if isinstance(model, PretrainedModel) and model.constructed_from_pretrained_config():
                    model.config.architectures = [model.__class__.__name__]
                    self.vdl_writer.add_text("model_config", str(model.config))
                elif hasattr(model, "init_config") and model.init_config is not None:
                    model_config_json = json.dumps(model.get_model_config(), ensure_ascii=False, indent=2)
                    self.vdl_writer.add_text("model_config", model_config_json)

            if hasattr(self.vdl_writer, "add_hparams"):
                self.vdl_writer.add_hparams(args.to_sanitized_dict(), metrics_list=[])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.vdl_writer is None:
            return

        if self.vdl_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.vdl_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of VisualDL's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.vdl_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.vdl_writer:
            self.vdl_writer.close()
            self.vdl_writer = None


class AutoNLPCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [`Ray Tune`] for [`AutoNLP`]
    """

    def __init__(self):
        if not is_ray_available():
            raise RuntimeError(
                "AutoNLPCallback requires extra dependencies to be installed. Please install paddlenlp with 'pip install paddlenlp[autonlp]'."
            )
        self.session = importlib.import_module("ray.air.session")
        self.tune = importlib.import_module("ray.tune")

    # report session metrics to Ray to track trial progress
    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        metrics = kwargs.get("metrics", None)
        if self.tune.is_session_enabled() and metrics is not None and isinstance(metrics, dict):
            self.session.report(metrics)


INTEGRATION_TO_CALLBACK = {
    "visualdl": VisualDLCallback,
    "autonlp": AutoNLPCallback,
}


def get_reporting_integration_callbacks(report_to):
    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )
    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
