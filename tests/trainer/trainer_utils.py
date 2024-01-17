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

import copy
import dataclasses
import random

import numpy as np
import paddle
import paddle.io as io
import paddle.nn as nn

from paddlenlp.trainer import Trainer, TrainingArguments
from paddlenlp.transformers import PretrainedConfig, PretrainedModel


def get_pretrain_arguments(pretrain_arguments):

    configs = {}

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 8
    train_args["pipeline_parallel_degree"] = 1
    configs["TP8"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 2
    train_args["pipeline_parallel_degree"] = 1
    configs["TP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 4
    train_args["pipeline_parallel_degree"] = 2
    configs["TP4PP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 4
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = ""
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    configs["TP4DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 4
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    configs["TP4Sharding2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 2
    train_args["pipeline_parallel_degree"] = 4
    configs["TP2PP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 2
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 4
    configs["TP2Sharding4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 8
    configs["PP8"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 4
    train_args["sharding"] = ""
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    configs["PP4DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 4
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    configs["PP4Sharding2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding8S1"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding8S2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 4
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding4S1DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 4
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding4S2DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 2
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding2S1DP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 2
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["Sharding2S2DP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 1
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    configs["DP8"] = train_args

    return configs


# Datasets for Test case
# RegressionDataset:
# RepeatDataset:
# DynamicShapesDataset:
# SampleIterableDataset:
# FiniteIterableDataset:
class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,)) for _ in self.label_names]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = {name: y[i] for name, y in zip(self.label_names, self.ys)}
        result["input_x"] = self.x[i]
        return result


class RepeatDataset:
    def __init__(self, x, length=64):
        self.x = x
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.x, "labels": self.x}


class DynamicShapesDataset:
    def __init__(self, length=64, seed=42, batch_size=8):
        self.length = length
        np.random.seed(seed)
        sizes = np.random.randint(1, 20, (length // batch_size,))
        # For easy batching, we make every batch_size consecutive samples the same size.
        self.xs = [np.random.normal(size=(s,)) for s in sizes.repeat(batch_size)]
        self.ys = [np.random.normal(size=(s,)) for s in sizes.repeat(batch_size)]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_x": self.xs[i], "labels": self.ys[i]}


class SampleIterableDataset(io.IterableDataset):
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        self.dataset = RegressionDataset(a=a, b=b, length=length, seed=seed, label_names=label_names)

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]


class FiniteIterableDataset(SampleIterableDataset):
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        super().__init__(a, b, length, seed, label_names)
        self.current_sample = 0

    def __iter__(self):
        while self.current_sample < len(self.dataset):
            yield self.dataset[self.current_sample]
            self.current_sample += 1


# args for Test case
# RegressionTrainingArguments:
@dataclasses.dataclass
class RegressionTrainingArguments(TrainingArguments):
    a: float = 0.0
    b: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        # save resources not dealing with reporting (also avoids the warning when it's not set)
        self.report_to = []


# config for Test case
# RegressionModelConfig
class RegressionModelConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, random_torch=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_torch = random_torch
        self.hidden_size = 1


# Models for Test case
# RegressionModel:
# RegressionPretrainedModel:
# RegressionRandomPretrainedModel:
# TstLayer:
class RegressionModel(nn.Layer):
    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.b = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.a.set_value(paddle.to_tensor(a, paddle.float32))
        self.b.set_value(paddle.to_tensor(b, paddle.float32))
        self.double_output = double_output
        self.config = None

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if labels is None:
            return (y, y) if self.double_output else (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y, y) if self.double_output else (loss, y)


class RegressionPretrainedModel(PretrainedModel):
    config_class = RegressionModelConfig
    base_model_prefix = "regression"

    def __init__(self, config):
        super().__init__(config)
        self.a = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.b = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.a.set_value(paddle.to_tensor(config.a, paddle.float32))
        self.b.set_value(paddle.to_tensor(config.b, paddle.float32))
        self.double_output = config.double_output

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if labels is None:
            return (y, y) if self.double_output else (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y, y) if self.double_output else (loss, y)


class RegressionRandomPretrainedModel(PretrainedModel):
    config_class = RegressionModelConfig
    base_model_prefix = "regression"

    def __init__(self, config):
        super().__init__(config)
        self.a = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.b = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.a.set_value(paddle.to_tensor(config.a, paddle.float32))
        self.b.set_value(paddle.to_tensor(config.b, paddle.float32))
        self.random_paddle = config.random_paddle

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if self.random_paddle:
            paddle_rand = paddle.randn(1).item()
        np_rand = np.random.rand()
        rand_rand = random.random()

        if self.random_paddle:
            y += 0.05 * paddle_rand
        y += 0.05 * paddle.to_tensor(np_rand + rand_rand)

        if labels is None:
            return (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y)


class RegressionDictModel(nn.Layer):
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.b = paddle.create_parameter(shape=[], dtype=paddle.float32)
        self.a.set_value(paddle.to_tensor(a, paddle.float32))
        self.b.set_value(paddle.to_tensor(b, paddle.float32))
        self.config = None

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        result = {"output": y}
        if labels is not None:
            result["loss"] = nn.functional.mse_loss(y, labels)
        return result


class TstLayer(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.bias = paddle.create_parameter(shape=hidden_size.shape)
        self.bias.set_value(paddle.zeros(hidden_size))

    def forward(self, x):
        h = self.ln1(nn.functional.relu(self.linear1(x)))
        h = nn.functional.relu(self.linear2(x))
        return self.ln2(x + h + self.bias)


class AlmostAccuracy:
    def __init__(self, thresh=0.25):
        self.thresh = thresh

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        true = np.abs(predictions - labels) <= self.thresh
        return {"accuracy": true.astype(np.float32).mean().item()}


class MultiLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self):
        for loader in self.loaders:
            yield from loader


class CustomDataloaderTrainer(Trainer):
    def get_train_dataloader(self):
        dataloaders = [super().get_train_dataloader(), super().get_train_dataloader()]
        return MultiLoader(dataloaders)

    def get_eval_dataloader(self, eval_dataset):
        dataloaders = [super().get_eval_dataloader(eval_dataset), super().get_eval_dataloader(eval_dataset)]
        return MultiLoader(dataloaders)


def get_regression_trainer(a=0, b=0, double_output=False, train_len=64, eval_len=64, pretrained=True, **kwargs):
    label_names = kwargs.get("label_names", None)
    train_dataset = RegressionDataset(length=train_len, label_names=label_names)
    eval_dataset = RegressionDataset(length=eval_len, label_names=label_names)

    if pretrained:
        config = RegressionModelConfig(a=a, b=b, double_output=double_output)
        model = RegressionPretrainedModel(config)
    else:
        model = RegressionModel(a=a, b=b, double_output=double_output)

    compute_metrics = kwargs.pop("compute_metrics", None)
    data_collator = kwargs.pop("data_collator", None)
    optimizers = kwargs.pop("optimizers", (None, None))
    output_dir = kwargs.pop("output_dir", "./regression")
    preprocess_logits_for_metrics = kwargs.pop("preprocess_logits_for_metrics", None)
    kwargs["keep_optimizer_state_static_keys"] = False

    args = RegressionTrainingArguments(output_dir, a=a, b=b, **kwargs)
    return Trainer(
        model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
