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
from typing import Any, Callable, Dict, List

import numpy as np
import paddle
from paddle.io import Dataset
from scipy.special import expit as sigmoid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.taskflow import Taskflow
from paddlenlp.trainer import CompressionArguments, Trainer, TrainingArguments
from paddlenlp.trainer.trainer_utils import EvalPrediction
from paddlenlp.transformers import (
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
            output_dir="trained_model",
            num_train_epochs=1,
            learning_rate=1e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            disable_tqdm=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )

    @property
    def _default_compress_argument(self) -> CompressionArguments:
        return CompressionArguments(
            width_mult_list=["3/4"],
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            output_dir="pruned_model",
        )

    @property
    def _model_candidates(self) -> List[Dict[str, Any]]:
        return [
            {
                "preset": "test",
                "language": "Chinese",
                "PreprocessArguments.max_length": 128,
                "TrainingArguments.per_device_train_batch_size": 2,
                "TrainingArguments.per_device_eval_batch_size": 2,
                "TrainingArguments.max_steps": 5,
                "TrainingArguments.model_name_or_path": "ernie-3.0-nano-zh",
                "TrainingArguments.learning_rate": 1e-5,
            }
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
            model_path = config["TrainingArguments.model_name_or_path"]
            max_length = config["PreprocessArguments.max_length"]
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            trans_func = functools.partial(
                self._preprocess_fn,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            processed_train_dataset = train_dataset.map(trans_func, lazy=False)
            processed_eval_dataset = eval_dataset.map(trans_func, lazy=False)

            # define model
            problem_type = (
                "multi_label_classification" if self.problem_type == "multi_label" else "single_label_classification"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_classes=len(self.id2label), problem_type=problem_type
            )
            training_args = self._override_training_arguments(config)
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                # criterion=paddle.nn.loss.CrossEntropyLoss(),
                train_dataset=processed_train_dataset,
                eval_dataset=processed_eval_dataset,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=self._compute_metrics,
            )
            trainer.train()
            trainer.save_model()
            eval_metrics = trainer.evaluate(eval_dataset)
            return eval_metrics

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
            if self.problem_type == "multi_class":
                result["labels"] = paddle.to_tensor([self.label2id[example[self.label_column]]], dtype="int64")
            # multi_label
            else:
                labels = [1.0 if i in example[self.label_column] else 0.0 for i in self.label2id]
                result["labels"] = paddle.to_tensor(labels, dtype="float32")
        return result

    def export(self, export_path, trial_id=None):
        model_result = self._get_model_result(trial_id=trial_id)
        saved_model_path = os.path.join(model_result.log_dir, "trained_model")
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(saved_model_path, num_classes=len(self.id2label))
        tokenizer.save_pretrained(export_path)
        model.save_pretrained(export_path)

    def to_taskflow(self, trial_id=None):
        model_result = self._get_model_result(trial_id=trial_id)
        model_config = model_result.metrics["config"]["candidates"]
        saved_model_path = os.path.join(model_result.log_dir, "trained_model")
        return Taskflow(
            "text_classification",
            task_path=saved_model_path,
            id2label=self.id2label,
            max_length=model_config["PreprocessArguments.max_length"],
        )
