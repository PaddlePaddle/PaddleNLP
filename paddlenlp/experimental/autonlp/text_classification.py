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
import functools
import os
import shutil
from asyncio.log import logger
from typing import Any, Callable, Dict, List

import numpy as np
import paddle
from hyperopt import hp
from paddle.io import Dataset
from scipy.special import expit as sigmoid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.prompt import (
    AutoTemplate,
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    SoftVerbalizer,
)
from paddlenlp.taskflow import Taskflow
from paddlenlp.trainer import EarlyStoppingCallback, Trainer, TrainingArguments
from paddlenlp.trainer.trainer_utils import EvalPrediction
from paddlenlp.transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedTokenizer,
)

from .auto_trainer_base import AutoTrainerBase


class AutoTrainerForTextClassification(AutoTrainerBase):
    """
    AutoTrainer for Text Classification problems

    Args:
        text_column (string, required): Name of the column that contains the input text.
        label_column (string, required): Name of the column that contains the target variable to predict.
        metric_for_best_model (string, optional): the name of the metrc for selecting the best model.
        greater_is_better (bool, optional): Whether better models should have a greater metric or not. Use in conjuction with `metric_for_best_model`.
        kwargs (dict, optional): Additional keyword arguments passed along to underlying meta class.
    """

    def __init__(
        self,
        text_column: str,
        label_column: str,
        metric_for_best_model: str = "eval_accuracy",
        greater_is_better: bool = True,
        problem_type: str = "multi_class",
        **kwargs
    ):

        super(AutoTrainerForTextClassification, self).__init__(
            metric_for_best_model=metric_for_best_model, greater_is_better=greater_is_better, **kwargs
        )
        self.text_column = text_column
        self.label_column = label_column
        if problem_type in ["multi_label", "multi_class"]:
            self.problem_type = problem_type
        else:
            raise ValueError(
                f"'{problem_type}' is not a supported problem_type. Please select among ['multi_label', 'multi_class']"
            )

    @property
    def _default_training_argument(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.training_path,
            num_train_epochs=1,
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            disable_tqdm=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
        )

    @property
    def _default_prompt_tuning_arguments(self) -> PromptTuningArguments:
        return PromptTuningArguments(
            output_dir=self.training_path,
            num_train_epochs=1,
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            disable_tqdm=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
        )

    @property
    def _model_candidates(self) -> List[Dict[str, Any]]:
        return [
            {
                "preset": "finetune",
                "language": "Chinese",
                "trainer_type": "Trainer",
                "PreprocessArguments.max_length": 128,
                "EarlyStoppingCallback.early_stopping_patience": 2,
                "TrainingArguments.per_device_train_batch_size": 16,
                "TrainingArguments.per_device_eval_batch_size": 16,
                "TrainingArguments.num_train_epochs": 10,
                "TrainingArguments.model_name_or_path": "ernie-3.0-base-zh",
                "TrainingArguments.learning_rate": 1e-5,
            },
            {
                "preset": "prompt",
                "language": "Chinese",
                "trainer_type": "PromptTrainer",
                "template.prompt": hp.choice(
                    "template.prompt",
                    [
                        f"{{'soft'}}{{'text': '{self.text_column}'}}{{'mask'}}",
                        f"“{{'text': '{self.text_column}'}}”这句话是关于{{'mask'}}的",
                    ],
                ),
                "PreprocessArguments.max_length": 128,
                "EarlyStoppingCallback.early_stopping_patience": 2,
                "PromptTuningArguments.per_device_train_batch_size": 16,
                "PromptTuningArguments.per_device_eval_batch_size": 16,
                "PromptTuningArguments.num_train_epochs": 10,
                "PromptTuningArguments.model_name_or_path": "ernie-3.0-base-zh",
                "PromptTuningArguments.learning_rate": 1e-5,
            },
            {
                "preset": "finetune_test",
                "language": "Chinese",
                "trainer_type": "Trainer",
                "TrainingArguments.max_steps": 5,
                "TrainingArguments.model_name_or_path": "ernie-3.0-nano-zh",
            },
            {
                "preset": "prompt_test",
                "language": "Chinese",
                "trainer_type": "PromptTrainer",
                "template.prompt": f"{{'soft'}}{{'text': '{self.text_column}'}}{{'mask'}}",
                "PromptTuningArguments.max_steps": 5,
                "PromptTuningArguments.model_name_or_path": "ernie-3.0-nano-zh",
            },
        ]

    def _data_checks_and_inference(self, train_dataset: Dataset, eval_dataset: Dataset):
        self.id2label, self.label2id = {}, {}
        # TODO: support label ids that is already encoded
        if self.problem_type == "multi_class":
            for dataset in [train_dataset, eval_dataset]:
                for example in dataset:
                    label = example[self.label_column]
                    if label not in self.label2id:
                        self.label2id[label] = len(self.label2id)
                        self.id2label[len(self.id2label)] = label
        # multi_label
        else:
            for dataset in [train_dataset, eval_dataset]:
                for example in dataset:
                    labels = example[self.label_column]
                    for label in labels:
                        if label not in self.label2id:
                            self.label2id[label] = len(self.label2id)
                            self.id2label[len(self.id2label)] = label

    def _construct_trainable(self, train_dataset: Dataset, eval_dataset: Dataset) -> Callable:
        def trainable(config):
            config = config["candidates"]
            max_length = config.get("PreprocessArguments.max_length", 128)
            if "EarlyStoppingCallback.early_stopping_patience" in config:
                callbacks = [
                    EarlyStoppingCallback(
                        early_stopping_patience=config["EarlyStoppingCallback.early_stopping_patience"]
                    )
                ]
            else:
                callbacks = None
            if config["trainer_type"] == "Trainer":
                model_path = config["TrainingArguments.model_name_or_path"]
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                trans_func = functools.partial(
                    self._preprocess_fn,
                    tokenizer=tokenizer,
                    max_length=max_length,
                )
                processed_train_dataset = train_dataset.map(trans_func, lazy=False)
                processed_eval_dataset = eval_dataset.map(trans_func, lazy=False)
                model = AutoModelForSequenceClassification.from_pretrained(model_path, num_classes=len(self.id2label))
                training_args = self._override_arguments(config, self._default_training_argument)
                trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    train_dataset=processed_train_dataset,
                    eval_dataset=processed_eval_dataset,
                    data_collator=DataCollatorWithPadding(tokenizer),
                    compute_metrics=self._compute_metrics,
                    callbacks=callbacks,
                )
                trainer.train()
                trainer.save_model(self.export_path)
                if os.path.exists(self.training_path):
                    logger.info("Removing training checkpoints to conserve disk space")
                    shutil.rmtree(self.training_path)
                eval_metrics = trainer.evaluate(eval_dataset)
                return eval_metrics
            elif config["trainer_type"] == "PromptTrainer":
                model_path = config["PromptTuningArguments.model_name_or_path"]
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                processed_train_dataset = train_dataset.map(self._preprocess_labels, lazy=False)
                processed_eval_dataset = eval_dataset.map(self._preprocess_labels, lazy=False)
                model = AutoModelForMaskedLM.from_pretrained(model_path)
                template = AutoTemplate.create_from(
                    prompt=config["template.prompt"],
                    tokenizer=tokenizer,
                    max_length=max_length,
                    model=model,
                )
                training_args = self._override_arguments(config, self._default_prompt_tuning_arguments)
                verbalizer = SoftVerbalizer(label_words=self.id2label, tokenizer=tokenizer, model=model)
                prompt_model = PromptModelForSequenceClassification(
                    model,
                    template,
                    verbalizer,
                    freeze_plm=training_args.freeze_plm,
                    freeze_dropout=training_args.freeze_dropout,
                )
                if self.problem_type == "multi_class":
                    criterion = paddle.nn.CrossEntropyLoss()
                else:  # multi_label
                    criterion = paddle.nn.BCEWithLogitsLoss()
                trainer = PromptTrainer(
                    model=prompt_model,
                    tokenizer=tokenizer,
                    args=training_args,
                    criterion=criterion,
                    train_dataset=processed_train_dataset,
                    eval_dataset=processed_eval_dataset,
                    callbacks=callbacks,
                    compute_metrics=self._compute_metrics,
                )
                trainer.train()
                trainer.save_model(self.export_path)
                if os.path.exists(self.training_path):
                    logger.info("Removing training checkpoints to conserve disk space")
                    shutil.rmtree(self.training_path)
                eval_metrics = trainer.evaluate(eval_dataset)
                return eval_metrics
            else:
                raise ValueError("'trainer_type' can only be one of ['Trainer', 'PromptTrainer']")

        return trainable

    def _compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        if self.problem_type == "multi_class":
            return self._compute_multi_class_metrics(eval_preds=eval_preds)
        else:  # multi_label
            return self._compute_multi_label_metrics(eval_preds=eval_preds)

    def _compute_multi_class_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true=eval_preds.label_ids, y_pred=pred_ids)
        for average in ["micro", "macro"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=eval_preds.label_ids, y_pred=pred_ids, average=average
            )
            metrics[f"{average}_precision"] = precision
            metrics[f"{average}_recall"] = recall
            metrics[f"{average}_f1"] = f1
        return metrics

    def _compute_multi_label_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        pred_probs = sigmoid(eval_preds.predictions)
        pred_ids = pred_probs > 0.5
        metrics = {}
        # In multilabel classification, this function computes subset accuracy:
        # the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
        metrics["accuracy"] = accuracy_score(y_true=eval_preds.label_ids, y_pred=pred_ids)
        for average in ["micro", "macro"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=eval_preds.label_ids, y_pred=pred_ids, average=average
            )
            metrics[f"{average}_precision"] = precision
            metrics[f"{average}_recall"] = recall
            metrics[f"{average}_f1"] = f1
        return metrics

    def _preprocess_labels(self, example):
        if self.problem_type == "multi_class":
            example["labels"] = paddle.to_tensor([self.label2id[example[self.label_column]]], dtype="int64")
        # multi_label
        else:
            labels = [1.0 if i in example[self.label_column] else 0.0 for i in self.label2id]
            example["labels"] = paddle.to_tensor(labels, dtype="float32")
        return example

    def _preprocess_fn(
        self,
        example: Dict[str, Any],
        tokenizer: PretrainedTokenizer,
        max_length: int,
        is_test: bool = False,
    ):
        """
        preprocess an example from raw features to input features that Transformers models expect (e.g. input_ids, attention_mask, labels, etc)
        """
        result = tokenizer(text=example[self.text_column], max_length=max_length)
        if not is_test:
            example_with_labels = self._preprocess_labels(example)
            result["labels"] = example_with_labels["labels"]
        return result

    def export(self, export_path, trial_id=None):
        model_result = self._get_model_result(trial_id=trial_id)
        exported_model_path = os.path.join(model_result.log_dir, self.export_path)
        shutil.copytree(exported_model_path, export_path)
        logger.info(f"Exported to {export_path}")

    def to_taskflow(self, trial_id=None):
        model_result = self._get_model_result(trial_id=trial_id)
        model_config = model_result.metrics["config"]["candidates"]
        if model_config["trainer_type"] == "PromptTrainer":
            raise NotImplementedError("'Taskflow' inference does not yet support models trained with PromptTrainer.")
        else:
            exported_model_path = os.path.join(model_result.log_dir, self.export_path)
            return Taskflow(
                "text_classification",
                task_path=exported_model_path,
                id2label=self.id2label,
                max_length=model_config.get("PreprocessArguments.max_length", 128),
            )
