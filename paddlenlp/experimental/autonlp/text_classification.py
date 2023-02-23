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
import functools
import json
import os
import shutil
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
    export_model,
)
from paddlenlp.utils.log import logger

from .auto_trainer_base import AutoTrainerBase


class AutoTrainerForTextClassification(AutoTrainerBase):
    """
    AutoTrainer for Text Classification problems

    Args:
        train_dataset (Dataset, required): Training dataset, must contains the 'text_column' and 'label_column' specified below
        eval_dataset (Dataset, required): Evaluation dataset, must contains the 'text_column' and 'label_column' specified below
        text_column (string, required): Name of the column that contains the input text.
        label_column (string, required): Name of the column that contains the target variable to predict.
        metric_for_best_model (string, optional): the name of the metrc for selecting the best model.
        greater_is_better (bool, optional): Whether better models should have a greater metric or not. Use in conjuction with `metric_for_best_model`.
        problem_type (str, optional): Select among ["multi_class", "multi_label"] based on the nature of your problem
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
            language (string, required): language of the text.
            output_dir (str, optional): Output directory for the experiments, defaults to "autpnlp_results".
            id2label(dict(int,string)): The dictionary to map the predictions from class ids to class names.
            multilabel_threshold (float): The probability threshold used for the multi_label setup. Only effective if model = "multi_label". Defaults to 0.5.
            verbosity: (int, optional): controls the verbosity of the run. Defaults to 1, which let the workers log to the driver.To reduce the amount of logs, use verbosity > 0 to set stop the workers from logging to the driver.
    """

    def __init__(
        self,
        text_column: str,
        label_column: str,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        metric_for_best_model: str = "eval_accuracy",
        greater_is_better: bool = True,
        problem_type: str = "multi_class",
        **kwargs
    ):

        super(AutoTrainerForTextClassification, self).__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            **kwargs,
        )
        self.text_column = text_column
        self.label_column = label_column
        self.id2label = self.kwargs.get("id2label", None)
        self.multilabel_threshold = self.kwargs.get("multilabel_threshold", 0.5)
        if problem_type in ["multi_label", "multi_class"]:
            self.problem_type = problem_type
        else:
            raise NotImplementedError(
                f"'{problem_type}' is not a supported problem_type. Please select among ['multi_label', 'multi_class']"
            )
        self._data_checks_and_inference()

    @property
    def supported_languages(self) -> List[str]:
        return ["Chinese", "English"]

    @property
    def _default_training_argument(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.training_path,
            disable_tqdm=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            report_to=["visualdl", "autonlp"],
        )

    @property
    def _default_prompt_tuning_arguments(self) -> PromptTuningArguments:
        return PromptTuningArguments(
            output_dir=self.training_path,
            disable_tqdm=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            report_to=["visualdl", "autonlp"],
        )

    @property
    def _model_candidates(self) -> List[Dict[str, Any]]:
        train_batch_size = hp.choice("batch_size", [2, 4, 8, 16, 32])
        chinese_finetune_models = hp.choice(
            "finetune_models",
            [
                "ernie-1.0-large-zh-cw",  # 24-layer, 1024-hidden, 16-heads, 272M parameters.
                "ernie-3.0-xbase-zh",  # 20-layer, 1024-hidden, 16-heads, 296M parameters.
                "ernie-3.0-tiny-base-v2-zh",  # 12-layer, 768-hidden, 12-heads, 118M parameters.
                "ernie-3.0-tiny-medium-v2-zh",  # 6-layer, 768-hidden, 12-heads, 75M parameters.
                "ernie-3.0-tiny-mini-v2-zh",  # 6-layer, 384-hidden, 12-heads, 27M parameters
                "ernie-3.0-tiny-micro-v2-zh",  # 4-layer, 384-hidden, 12-heads, 23M parameters
                "ernie-3.0-tiny-nano-v2-zh",  # 4-layer, 312-hidden, 12-heads, 18M parameters.
                "ernie-3.0-tiny-pico-v2-zh",  # 3-layer, 128-hidden, 2-heads, 5.9M parameters.
            ],
        )
        english_finetune_models = hp.choice(
            "finetune_models",
            [
                # add deberta-v3 when we have it
                "roberta-large",  # 24-layer, 1024-hidden, 16-heads, 334M parameters. Case-sensitive
                "roberta-base",  # 12-layer, 768-hidden, 12-heads, 110M parameters. Case-sensitive
                "distilroberta-base",  # 6-layer, 768-hidden, 12-heads, 66M parameters. Case-sensitive
                "ernie-2.0-base-en",  # 12-layer, 768-hidden, 12-heads, 103M parameters. Trained on lower-cased English text.
                "ernie-2.0-large-en",  # 24-layer, 1024-hidden, 16-heads, 336M parameters. Trained on lower-cased English text.
            ],
        )
        english_prompt_models = hp.choice(
            "prompt_models",
            [
                # add deberta-v3 when we have it
                "roberta-large",  # 24-layer, 1024-hidden, 16-heads, 334M parameters. Case-sensitive
                "roberta-base",  # 12-layer, 768-hidden, 12-heads, 110M parameters. Case-sensitive
            ],
        )
        chinese_prompt_models = hp.choice(
            "prompt_models",
            [
                "ernie-1.0-large-zh-cw",  # 24-layer, 1024-hidden, 16-heads, 272M parameters.
                "ernie-1.0-base-zh-cw",  # 12-layer, 768-hidden, 12-heads, 118M parameters.
            ],
        )
        return [
            # fast learning: high LR, small early stop patience
            {
                "preset": "finetune",
                "language": "Chinese",
                "trainer_type": "Trainer",
                "EarlyStoppingCallback.early_stopping_patience": 5,
                "TrainingArguments.per_device_train_batch_size": train_batch_size,
                "TrainingArguments.per_device_eval_batch_size": train_batch_size * 2,
                "TrainingArguments.num_train_epochs": 100,
                "TrainingArguments.model_name_or_path": chinese_finetune_models,
                "TrainingArguments.learning_rate": 3e-5,
            },
            {
                "preset": "finetune",
                "language": "English",
                "trainer_type": "Trainer",
                "EarlyStoppingCallback.early_stopping_patience": 5,
                "TrainingArguments.per_device_train_batch_size": train_batch_size,
                "TrainingArguments.per_device_eval_batch_size": train_batch_size * 2,
                "TrainingArguments.num_train_epochs": 100,
                "TrainingArguments.model_name_or_path": english_finetune_models,
                "TrainingArguments.learning_rate": 3e-5,
            },
            # slow learning: small LR, large early stop patience
            {
                "preset": "finetune",
                "language": "Chinese",
                "trainer_type": "Trainer",
                "EarlyStoppingCallback.early_stopping_patience": 5,
                "TrainingArguments.per_device_train_batch_size": train_batch_size,
                "TrainingArguments.per_device_eval_batch_size": train_batch_size * 2,
                "TrainingArguments.num_train_epochs": 100,
                "TrainingArguments.model_name_or_path": chinese_finetune_models,
                "TrainingArguments.learning_rate": 5e-6,
            },
            {
                "preset": "finetune",
                "language": "English",
                "trainer_type": "Trainer",
                "EarlyStoppingCallback.early_stopping_patience": 5,
                "TrainingArguments.per_device_train_batch_size": train_batch_size,
                "TrainingArguments.per_device_eval_batch_size": train_batch_size * 2,
                "TrainingArguments.num_train_epochs": 100,
                "TrainingArguments.model_name_or_path": english_finetune_models,
                "TrainingArguments.learning_rate": 5e-6,
            },
            # prompt tuning candidates
            {
                "preset": "prompt",
                "language": "Chinese",
                "trainer_type": "PromptTrainer",
                "template.prompt": "{'mask'}{'soft'}“{'text': '" + self.text_column + "'}”",
                "EarlyStoppingCallback.early_stopping_patience": 5,
                "PromptTuningArguments.per_device_train_batch_size": train_batch_size,
                "PromptTuningArguments.per_device_eval_batch_size": train_batch_size * 2,
                "PromptTuningArguments.num_train_epochs": 100,
                "PromptTuningArguments.model_name_or_path": chinese_prompt_models,
                "PromptTuningArguments.learning_rate": 1e-5,
                "PromptTuningArguments.ppt_learning_rate": 1e-4,
            },
            {
                "preset": "prompt",
                "language": "English",
                "trainer_type": "PromptTrainer",
                "template.prompt": "{'mask'}{'soft'}“{'text': '" + self.text_column + "'}”",
                "EarlyStoppingCallback.early_stopping_patience": 5,
                "PromptTuningArguments.per_device_train_batch_size": train_batch_size,
                "PromptTuningArguments.per_device_eval_batch_size": train_batch_size * 2,
                "PromptTuningArguments.num_train_epochs": 100,
                "PromptTuningArguments.model_name_or_path": english_prompt_models,
                "PromptTuningArguments.learning_rate": 1e-5,
                "PromptTuningArguments.ppt_learning_rate": 1e-4,
            },
        ]

    def _data_checks_and_inference(self):
        if self.id2label is None:
            self.id2label, self.label2id = {}, {}
            if self.problem_type == "multi_class":
                for dataset in [self.train_dataset, self.eval_dataset]:
                    for example in dataset:
                        label = example[self.label_column]
                        if label not in self.label2id:
                            self.label2id[label] = len(self.label2id)
                            self.id2label[len(self.id2label)] = label
            # multi_label
            else:
                for dataset in [self.train_dataset, self.eval_dataset]:
                    for example in dataset:
                        labels = example[self.label_column]
                        for label in labels:
                            if label not in self.label2id:
                                self.label2id[label] = len(self.label2id)
                                self.id2label[len(self.id2label)] = label
        else:
            self.label2id = {}
            for i in self.id2label:
                self.label2id[self.id2label[i]] = i

            if self.problem_type == "multi_class":
                for dataset in [self.train_dataset, self.eval_dataset]:
                    for example in dataset:
                        label = example[self.label_column]
                        if label not in self.label2id:
                            raise ValueError(
                                f"Label {label} is not found in the user-provided id2label argument: {self.id2label}"
                            )
            # multi_label
            else:
                for dataset in [self.train_dataset, self.eval_dataset]:
                    for example in dataset:
                        labels = example[self.label_column]
                        for label in labels:
                            if label not in self.label2id:
                                raise ValueError(
                                    f"Label {label} is not found in the user-provided id2label argument: {self.id2label}"
                                )

    def _construct_trainer(self, config, eval_dataset=None) -> Trainer:
        if "EarlyStoppingCallback.early_stopping_patience" in config:
            callbacks = [
                EarlyStoppingCallback(early_stopping_patience=config["EarlyStoppingCallback.early_stopping_patience"])
            ]
        else:
            callbacks = None
        if config["trainer_type"] == "Trainer":
            model_path = config["TrainingArguments.model_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=len(self.id2label), id2label=self.id2label, label2id=self.label2id
            )
            max_length = config.get("PreprocessArguments.max_length", model.config.max_position_embeddings)
            trans_func = functools.partial(
                self._preprocess_fn,
                tokenizer=tokenizer,
                max_length=max_length,  # truncate to the max length allowed by the model
            )
            processed_train_dataset = copy.deepcopy(self.train_dataset).map(trans_func, lazy=False)
            if eval_dataset is None:
                processed_eval_dataset = copy.deepcopy(self.eval_dataset).map(trans_func, lazy=False)
            else:
                processed_eval_dataset = copy.deepcopy(eval_dataset).map(trans_func, lazy=False)
            training_args = self._override_hp(config, self._default_training_argument)
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
        elif config["trainer_type"] == "PromptTrainer":
            model_path = config["PromptTuningArguments.model_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            processed_train_dataset = copy.deepcopy(self.train_dataset).map(self._preprocess_labels, lazy=False)
            if eval_dataset is None:
                processed_eval_dataset = copy.deepcopy(self.eval_dataset).map(self._preprocess_labels, lazy=False)
            else:
                processed_eval_dataset = copy.deepcopy(eval_dataset).map(self._preprocess_labels, lazy=False)

            model = AutoModelForMaskedLM.from_pretrained(model_path)
            max_length = config.get("PreprocessArguments.max_length", model.config.max_position_embeddings)
            template = AutoTemplate.create_from(
                prompt=config["template.prompt"],
                tokenizer=tokenizer,
                max_length=max_length,
                model=model,
            )
            training_args = self._override_hp(config, self._default_prompt_tuning_arguments)
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
        else:
            raise NotImplementedError("'trainer_type' can only be one of ['Trainer', 'PromptTrainer']")
        return trainer

    def _construct_trainable(self) -> Callable:
        def trainable(config):
            # import is required for proper pickling
            from paddlenlp.utils.log import logger

            config = config["candidates"]
            trainer = self._construct_trainer(config)
            trainer.train()
            eval_metrics = trainer.evaluate()
            trainer.save_model(self.save_path)

            if os.path.exists(self.training_path):
                logger.info("Removing training checkpoints to conserve disk space")
                shutil.rmtree(self.training_path)
            return eval_metrics

        return trainable

    def evaluate(self, trial_id=None, eval_dataset=None):
        """
        Evaluate the models from a certain `trial_id` on the given dataset

        Args:
            trial_id (str, optional): specify the model to be evaluated through the `trial_id`. Defaults to the best model selected by `metric_for_best_model`
            eval_dataset (Dataset, optional): custom evaluation dataset and must contains the 'text_column' and 'label_column' fields.
                If not provided, defaults to the evaluation dataset used at construction
        """
        model_result = self._get_model_result(trial_id=trial_id)
        model_config = model_result.metrics["config"]["candidates"]

        trainer = self._construct_trainer(model_config, eval_dataset)
        trainer.load_state_dict_from_checkpoint(
            resume_from_checkpoint=os.path.join(model_result.log_dir, self.save_path)
        )

        eval_metrics = trainer.evaluate()
        if os.path.exists(self.training_path):
            logger.info(f"Removing {self.training_path} to conserve disk space")
            shutil.rmtree(self.training_path)
        trainer.log_metrics("eval", eval_metrics)
        return eval_metrics

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
        pred_ids = pred_probs > self.multilabel_threshold
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
        result = tokenizer(text=example[self.text_column], max_length=max_length, truncation=True)
        if not is_test:
            example_with_labels = self._preprocess_labels(example)
            result["labels"] = example_with_labels["labels"]
        return result

    def to_taskflow(self, trial_id=None, export_path=None, batch_size=1, precision="fp32"):
        """
        Convert the model from a certain `trial_id` to a Taskflow for model inference

        Args:
            trial_id (int): use the `trial_id` to select the model to export. Defaults to the best model selected by `metric_for_best_model`
            export_path (str): the filepath to export to.
            batch_size(int): The sample number of a mini-batch. Defaults to 1.
            precision (str): Select among ["fp32", "fp16"]. Default to "fp32".
        """
        model_result = self._get_model_result(trial_id=trial_id)
        trial_id = model_result.metrics["trial_id"]

        if export_path is None:
            export_path = os.path.join(self.export_path, trial_id)

        taskflow_config = self.export(export_path=export_path, trial_id=trial_id)
        taskflow_config["task_path"] = export_path
        taskflow_config["batch_size"] = batch_size
        taskflow_config["precision"] = precision
        return Taskflow(**taskflow_config)

    def export(self, export_path, trial_id=None):
        """
        Export the model from a certain `trial_id` to the given file path.

        Args:
            export_path (str, required): the filepath to export to
            trial_id (int, required): use the `trial_id` to select the model to export. Defaults to the best model selected by `metric_for_best_model`
        """

        model_result = self._get_model_result(trial_id=trial_id)
        model_config = model_result.metrics["config"]["candidates"]
        trial_id = model_result.metrics["trial_id"]

        if os.path.exists(export_path):
            logger.info(
                f"Export path for {trial_id} already exists: ({export_path}). The model parameter files will be overwritten."
            )

        # construct trainer
        trainer = self._construct_trainer(model_config)
        trainer.load_state_dict_from_checkpoint(
            resume_from_checkpoint=os.path.join(model_result.log_dir, self.save_path)
        )

        # save static model
        if model_config["trainer_type"] == "PromptTrainer":
            trainer.export_model(export_path)
            trainer.model.plm.save_pretrained(os.path.join(export_path, "plm"))
            mode = "prompt"
            max_length = model_config.get(
                "PreprocessArguments.max_length", trainer.model.plm.config.max_position_embeddings
            )
        else:
            if trainer.model.init_config["init_class"] in ["ErnieMForSequenceClassification"]:
                input_spec = [paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids")]
            else:
                input_spec = [
                    paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                    paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
                ]
            export_model(model=trainer.model, input_spec=input_spec, path=export_path)
            mode = "finetune"
            max_length = model_config.get(
                "PreprocessArguments.max_length", trainer.model.config.max_position_embeddings
            )

        # save tokenizer
        trainer.tokenizer.save_pretrained(export_path)

        # save taskflow config file
        taskflow_config = {
            "task": "text_classification",
            "mode": mode,
            "is_static_model": True,
            "problem_type": self.problem_type,
            "multilabel_threshold": self.multilabel_threshold,
            "max_length": max_length,
            "id2label": self.id2label,
        }

        with open(os.path.join(export_path, "taskflow_config.json"), "w", encoding="utf-8") as f:
            json.dump(taskflow_config, f, ensure_ascii=False)
        logger.info(
            f"Taskflow config saved to {export_path}. You can use the Taskflow config to create a Taskflow instance for inference"
        )

        if os.path.exists(self.training_path):
            logger.info("Removing training checkpoints to conserve disk space")
            shutil.rmtree(self.training_path)

        logger.info(f"Exported {trial_id} to {export_path}")
        return taskflow_config
