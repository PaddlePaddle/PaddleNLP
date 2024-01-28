# Copyright 2018 the HuggingFace Inc. team.
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

import dataclasses
import json
import os
import random
import re
import tempfile
from pathlib import Path

import numpy as np

from paddlenlp.trainer import IntervalStrategy, TrainingArguments
from paddlenlp.trainer.trainer_utils import PREFIX_CHECKPOINT_DIR

# from paddlenlp.trainer.training_args import OptimizerNames
from paddlenlp.transformers import PretrainedConfig
from paddlenlp.transformers.testing_utils import (
    TestCasePlus,
    get_gpu_count,
    get_tests_dir,
    require_paddle,
    require_sentencepiece,
)
from paddlenlp.utils.import_utils import is_paddle_available

# from paddlenlp.transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME

WEIGHTS_NAME = "model_state.pdparams"
WEIGHTS_INDEX_NAME = "model_state.pdparams.index.json"

if is_paddle_available():
    import paddle
    from paddle import nn
    from paddle.io import IterableDataset

    from paddlenlp.trainer import EarlyStoppingCallback, Trainer, TrainerState
    from paddlenlp.transformers import PretrainedModel

    # from paddlenlp.transformers.model_utils import unwrap_model


PATH_SAMPLE_TEXT = f"{get_tests_dir()}/fixtures/sample_text.txt"


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


@dataclasses.dataclass
class RegressionTrainingArguments(TrainingArguments):
    a: float = 0.0
    b: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        # save resources not dealing with reporting (also avoids the warning when it's not set)
        self.report_to = []


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


class AlmostAccuracy:
    def __init__(self, thresh=0.25):
        self.thresh = thresh

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        true = np.abs(predictions - labels) <= self.thresh
        return {"accuracy": true.astype(np.float32).mean().item()}


class RegressionModelConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, random_paddle=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_paddle = random_paddle
        self.hidden_size = 1


class SampleIterableDataset(IterableDataset):
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


class TrainerIntegrationCommon:
    def check_saved_checkpoints(self, output_dir, freq, total, is_pretrained=True):
        file_list = [WEIGHTS_NAME, "training_args.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
        if is_pretrained:
            file_list.append("config.json")
        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint))
            for filename in file_list:
                self.assertTrue(os.path.isfile(os.path.join(checkpoint, filename)))

    def check_best_model_has_been_loaded(
        self, output_dir, freq, total, trainer, metric, greater_is_better=False, is_pretrained=True
    ):
        checkpoint = os.path.join(output_dir, f"checkpoint-{(total // freq) * freq}")
        log_history = TrainerState.load_from_json(os.path.join(checkpoint, "trainer_state.json")).log_history

        values = [d[metric] for d in log_history]
        best_value = max(values) if greater_is_better else min(values)
        best_checkpoint = (values.index(best_value) + 1) * freq
        checkpoint = os.path.join(output_dir, f"checkpoint-{best_checkpoint}")
        if is_pretrained:
            best_model = RegressionPretrainedModel.from_pretrained(checkpoint)
            best_model.to(trainer.args.device)
        else:
            best_model = RegressionModel()
            state_dict = paddle.load(os.path.join(checkpoint, WEIGHTS_NAME))
            best_model.load_state_dict(state_dict)
            best_model.to(trainer.args.device)
        self.assertTrue(paddle.allclose(best_model.a, trainer.model.a))
        self.assertTrue(paddle.allclose(best_model.b, trainer.model.b))

        metrics = trainer.evaluate()
        self.assertEqual(metrics[metric], best_value)

    def check_trainer_state_are_the_same(self, trainer_state, trainer_state1):
        # We'll pop things so operate on copies.
        state = trainer_state.copy()
        state1 = trainer_state1.copy()
        # Log history main contain different logs for the time metrics (after resuming a training).
        log_history = state.pop("log_history", None)
        log_history1 = state1.pop("log_history", None)
        self.assertEqual(state, state1)
        skip_log_keys = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss"]
        for log, log1 in zip(log_history, log_history1):
            for key in skip_log_keys:
                _ = log.pop(key, None)
                _ = log1.pop(key, None)
            self.assertEqual(log, log1)

    def convert_to_sharded_checkpoint(self, folder):
        # Converts a checkpoint of a regression model to a sharded checkpoint.
        state_dict = paddle.load(os.path.join(folder, WEIGHTS_NAME))
        os.remove(os.path.join(folder, WEIGHTS_NAME))
        keys = list(state_dict.keys())

        shard_files = [
            WEIGHTS_NAME.replace(".bin", f"-{idx+1:05d}-of-{len(keys):05d}.bin") for idx in range(len(keys))
        ]
        index = {"metadata": {}, "weight_map": {key: shard_files[i] for i, key in enumerate(keys)}}

        save_index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        for param_name, shard_file in zip(keys, shard_files):
            paddle.save({param_name: state_dict[param_name]}, os.path.join(folder, shard_file))


@require_paddle
@require_sentencepiece
class TrainerIntegrationPrerunTest(TestCasePlus, TrainerIntegrationCommon):
    """
    Only tests that want to tap into the auto-pre-run 2 trainings:
    - self.default_trained_model
    - self.alternate_trained_model
    directly, or via check_trained_model
    """

    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size
        trainer = get_regression_trainer(learning_rate=0.1)
        trainer.train()
        self.default_trained_model = (trainer.model.a, trainer.model.b)

        trainer = get_regression_trainer(learning_rate=0.1, seed=314)
        trainer.train()
        self.alternate_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, alternate_seed=False):
        # Checks a training seeded with learning_rate = 0.1
        (a, b) = self.alternate_trained_model if alternate_seed else self.default_trained_model
        self.assertTrue(paddle.allclose(model.a, a))
        self.assertTrue(paddle.allclose(model.b, b))

    def test_reproducible_training(self):
        # Checks that training worked, model trained and seed made a reproducible training.
        trainer = get_regression_trainer(learning_rate=0.1)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Checks that a different seed gets different (reproducible) results.
        trainer = get_regression_trainer(learning_rate=0.1, seed=314)
        trainer.train()
        self.check_trained_model(trainer.model, alternate_seed=True)

    def test_trainer_with_datasets(self):
        import datasets

        np.random.seed(42)
        x = np.random.normal(size=(64,)).astype(np.float32)
        y = 2.0 * x + 3.0 + np.random.normal(scale=0.1, size=(64,))
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y})

        # Base training. Should have the same results as test_reproducible_training
        model = RegressionModel()
        args = TrainingArguments("./regression", learning_rate=0.1)
        trainer = Trainer(model, args=args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Can return tensors.
        train_dataset.set_format(type="np", dtype=np.float32)
        model = RegressionModel()
        trainer = Trainer(model, args=args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Adding one column not used by the model should have no impact
        z = np.random.normal(size=(64,)).astype(np.float32)
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y, "extra": z})
        model = RegressionModel()
        trainer = Trainer(model, args=args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

    def test_gradient_accumulation(self):
        # Training with half the batch size but accumulation steps as 2 should give the same results.
        trainer = get_regression_trainer(
            gradient_accumulation_steps=2, per_device_train_batch_size=4, learning_rate=0.1
        )
        trainer.train()
        self.check_trained_model(trainer.model)

    def test_training_loss(self):
        n_gpus = max(1, get_gpu_count())

        # With even logs
        trainer = get_regression_trainer(logging_steps=64 / (8 * n_gpus))
        trainer.train()
        log_history = trainer.state.log_history

        losses = [log["loss"] for log in log_history if "loss" in log]
        train_loss = log_history[-1]["train_loss"]
        self.assertAlmostEqual(sum(losses) / len(losses), train_loss, places=4)

        # With uneven logs
        trainer = get_regression_trainer(logging_steps=5)
        trainer.train()
        log_history = trainer.state.log_history

        # Training loss should be the same as before
        new_train_loss = log_history[-1]["train_loss"]
        self.assertAlmostEqual(train_loss, new_train_loss, places=4)

    def test_custom_optimizer(self):
        train_dataset = RegressionDataset()
        args = TrainingArguments("./regression")
        model = RegressionModel()
        lr_scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate=1.0, lr_lambda=lambda x: 1.0)
        optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=lr_scheduler)
        trainer = Trainer(model, args=args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler))
        trainer.train()

        (a, b) = self.default_trained_model
        self.assertFalse(paddle.allclose(trainer.model.a, a))
        self.assertFalse(paddle.allclose(trainer.model.b, b))
        # self.assertEqual(trainer.optimizer.state_dict()["param_groups"][0]["lr"], 1.0)


@require_paddle
@require_sentencepiece
class TrainerIntegrationTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_trainer_works_with_dict(self):
        # Edge case because Apex with mode O2 will change our models to return dicts. This test checks it doesn't break
        # anything.
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionDictModel()
        args = TrainingArguments("./regression")
        trainer = Trainer(model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()
        _ = trainer.evaluate()
        _ = trainer.predict(eval_dataset)

    def test_number_of_steps_in_training(self):
        # Regular training has n_epochs * len(train_dl) steps
        trainer = get_regression_trainer(learning_rate=0.1)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, self.n_epochs * 64 / self.batch_size)

        # Check passing num_train_epochs works (and a float version too):
        trainer = get_regression_trainer(learning_rate=0.1, num_train_epochs=1.5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(1.5 * 64 / self.batch_size))

        # If we pass a max_steps, num_train_epochs is ignored
        trainer = get_regression_trainer(learning_rate=0.1, max_steps=10)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 10)

    def test_evaluate(self):
        trainer = get_regression_trainer(a=1.5, b=2.5, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With logits preprocess
        trainer = get_regression_trainer(
            a=1.5,
            b=2.5,
            compute_metrics=AlmostAccuracy(),
            preprocess_logits_for_metrics=lambda logits, labels: logits + 1,
        )
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred + 1, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_training_iterable_dataset(self):
        config = RegressionModelConfig()
        model = RegressionPretrainedModel(config)
        # Adding one column not used by the model should have no impact
        train_dataset = SampleIterableDataset(label_names=["labels", "extra"])

        args = RegressionTrainingArguments(output_dir="./examples", max_steps=4)
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
        trainer.train()
        self.assertEqual(trainer.state.global_step, 4)

        loader = trainer.get_train_dataloader()
        self.assertIsInstance(loader, paddle.io.DataLoader)
        self.assertIsInstance(loader.batch_sampler, paddle.io.dataloader.batch_sampler._InfiniteIterableSampler)

    def test_evaluation_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPretrainedModel(config)
        # Adding one column not used by the model should have no impact
        eval_dataset = SampleIterableDataset(label_names=["labels", "extra"])

        args = RegressionTrainingArguments(output_dir="./examples")
        trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.dataset.x, trainer.eval_dataset.dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        eval_dataset = SampleIterableDataset(length=66)
        results = trainer.evaluate(eval_dataset)

        x, y = eval_dataset.dataset.x, eval_dataset.dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_num_train_epochs_in_training(self):
        # len(train_dl) < gradient_accumulation_steps shouldn't give ``ZeroDivisionError`` when ``max_steps`` is given.
        # It should give 1 update step for each epoch.
        trainer = get_regression_trainer(
            max_steps=3, train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5
        )
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 3)

        # Even ``max_steps`` is not specified, we still expect 1 update step for each epoch if
        # len(train_dl) < gradient_accumulation_steps.
        trainer = get_regression_trainer(train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(self.n_epochs))

    def test_early_stopping_callback(self):
        # early stopping stops training before num_training_epochs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                load_best_model_at_end=True,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1, 0.0001))
            train_output = trainer.train()
            self.assertLess(train_output.global_step, 20 * 64 / 16)

        # Invalid inputs to trainer with early stopping callback result in assertion error
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                evaluation_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1))
            self.assertEqual(trainer.state.global_step, 0)
            try:
                trainer.train()
            except AssertionError:
                self.assertEqual(trainer.state.global_step, 0)

    def check_checkpoint_deletion(self, trainer, output_dir, expected):
        # Make fake checkpoints
        for n in [5, 10, 15, 20, 25]:
            os.makedirs(os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{n}"), exist_ok=True)
        trainer._rotate_checkpoints(output_dir=output_dir)
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")]
        values = [int(re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", d).groups()[0]) for d in glob_checkpoints]
        self.assertSetEqual(set(values), set(expected))

    def test_checkpoint_rotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Without best model at end
            trainer = get_regression_trainer(output_dir=tmp_dir, save_total_limit=2)
            self.check_checkpoint_deletion(trainer, tmp_dir, [20, 25])

            # With best model at end
            trainer = get_regression_trainer(
                output_dir=tmp_dir, evaluation_strategy="steps", load_best_model_at_end=True, save_total_limit=2
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

            # Edge case: we don't always honor save_total_limit=1 if load_best_model_at_end=True to be able to resume
            # from checkpoint
            trainer = get_regression_trainer(
                output_dir=tmp_dir, evaluation_strategy="steps", load_best_model_at_end=True, save_total_limit=1
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-25")
            self.check_checkpoint_deletion(trainer, tmp_dir, [25])

            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])
