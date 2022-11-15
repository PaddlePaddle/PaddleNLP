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
import copy
import os
import functools
from typing import Any, Callable, Dict, List

import paddle
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from hyperopt import hp
from paddle.io import Dataset
from paddle.metric import Accuracy

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.trainer import CompressionArguments, Trainer, TrainingArguments
from paddlenlp.trainer.trainer_utils import EvalPrediction
from paddlenlp.transformers import (AutoModelForSequenceClassification,
                                    AutoTokenizer, PretrainedTokenizer)

from .auto_trainer_base import AutoTrainerBase


class AutoTrainerForTextClassification(AutoTrainerBase):
    """
    AutoTrainer for Text Classification problems

    Args:
        text_column (string, required): Name of the column that contains the input text.
        label_column (string, required): Name of the column that contains the target variable to predict.
        metric_for_best_model (string, optional): the name of the metrc for selecting the best model
        kwargs (dict, optional): Additional keyword arguments passed along to underlying meta class. 
    """

    def __init__(self,
                 text_column: str,
                 label_column: str,
                 metric_for_best_model: str = "eval_accuracy",
                 greater_is_better: bool = True,
                 **kwargs):

        super(AutoTrainerForTextClassification,
              self).__init__(metric_for_best_model=metric_for_best_model,
                             greater_is_better=greater_is_better,
                             **kwargs)
        self.text_column = text_column
        self.label_column = label_column

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
        return [{
            "preset": "test",
            "language": "Chinese",
            "PreprocessArguments.max_length": 128,
            "TrainingArguments.per_device_train_batch_size": 2,
            "TrainingArguments.per_device_eval_batch_size": 2,
            "TrainingArguments.max_steps": 5,
            "TrainingArguments.model_name_or_path": "ernie-3.0-nano-zh",
            "TrainingArguments.learning_rate": 1e-5,
        }]

    def _data_checks_and_inference(self, train_dataset: Dataset,
                                   eval_dataset: Dataset):
        # TODO: support label ids that is already encoded
        train_labels = {i[self.label_column] for i in train_dataset}
        dev_labels = {i[self.label_column] for i in eval_dataset}
        self.id2label = list(train_labels.union(dev_labels))
        self.label2id = {label: i for i, label in enumerate(self.id2label)}

    # def _construct_trainer(self)

    def _construct_trainable(self, train_dataset: Dataset,
                             eval_dataset: Dataset) -> Callable:

        def trainable(config):
            config = config["config"]
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
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_classes=len(self.id2label))
            training_args = self._override_training_arguments(config)
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                criterion=paddle.nn.loss.CrossEntropyLoss(),
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
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true=eval_preds.label_ids,
                y_pred=pred_ids)
        for average in ['micro', 'macro']:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=eval_preds.label_ids,
                y_pred=pred_ids,
                average=average)
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
        result = tokenizer(text=example[self.text_column],
                           max_length=max_length)
        if not is_test:
            result["labels"] = paddle.to_tensor(
                [self.label2id[example[self.label_column]]], dtype="int64")
        return result

    def predict(self, test_dataset, trial_id=None) -> Dataset:
        model_result = self._get_model_result(trial_id=trial_id)
        model_config = model_result.metrics["config"]["config"]
        saved_model_path = os.path.join(model_result.log_dir, "trained_model")
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            saved_model_path, num_classes=len(self.id2label))
        trans_func = functools.partial(
            self._preprocess_fn,
            tokenizer=tokenizer,
            max_length=model_config["PreprocessArguments.max_length"],
            is_test=True)
        # since dataset.map modifies the underlying dataset in-place, create a deepcopy
        local_test_dataset = copy.deepcopy(test_dataset)
        processed_test_dataset = local_test_dataset.map(trans_func, lazy=False)
        training_args = self._override_training_arguments(model_config)
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            criterion=paddle.nn.loss.CrossEntropyLoss(),
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=self._compute_metrics,
        )
        test_results = trainer.predict(processed_test_dataset)
        return test_results

    def export(self, export_path, trial_id=None):
        model_result = self._get_model_result(trial_id=trial_id)
        saved_model_path = os.path.join(model_result.log_dir, "trained_model")
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            saved_model_path, num_classes=len(self.id2label))
        tokenizer.save_pretrained(export_path)
        model.save_pretrained(export_path)
