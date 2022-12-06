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
import datetime
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from hyperopt import hp
from paddle.io import Dataset
from ray import tune
from ray.air import RunConfig
from ray.tune.result_grid import ResultGrid
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

from paddlenlp.trainer import TrainingArguments
from paddlenlp.trainer.trainer_utils import EvalPrediction
from paddlenlp.transformers import PretrainedTokenizer


class AutoTrainerBase(metaclass=ABCMeta):
    """
    The meta classs of AutoTrainer, which contains the common properies and methods of AutoNLP.
    Task-specific AutoTrainers need to inherit from the meta class.

    Args:
        language (string, optional): language of the text
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(
        self,
        metric_for_best_model: str,
        greater_is_better: bool,
        language: str,
        output_dir: str = None,
        **kwargs,
    ):
        if not metric_for_best_model.startswith("eval_"):
            self.metric_for_best_model = f"eval_{metric_for_best_model}"
        else:
            self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.language = language
        self.output_dir = output_dir

    @property
    @abstractmethod
    def _default_training_argument(self) -> TrainingArguments:
        """
        Default TrainingArguments for the Trainer
        """

    @property
    @abstractmethod
    def _model_candidates(self) -> List[Dict[str, Any]]:
        """
        Model Candidates stored as Ray hyperparameter search space, organized by
        self.language and preset
        """

    @abstractmethod
    def _data_checks_and_inference(self, train_dataset: Dataset, eval_dataset: Dataset):
        """
        Performs different data checks and inferences on the training and eval datasets
        """

    @abstractmethod
    def _construct_trainable(self, train_dataset: Dataset, eval_dataset: Dataset) -> Callable:
        """
        Returns the Trainable functions that contains the main preprocessing and training logic
        """

    @abstractmethod
    def _compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        """
        function used by the Trainer to compute metrics during training
        See :class:`~paddlenlp.trainer.trainer_base.Trainer` for more details.
        """

    @abstractmethod
    def _preprocess_fn(
        self,
        example: Dict[str, Any],
        tokenizer: PretrainedTokenizer,
        max_seq_length: int,
        is_test: bool = False,
    ) -> Dict[str, Any]:
        """
        preprocess an example from raw features to input features that Transformers models expect (e.g. input_ids, attention_mask, labels, etc)
        """

    @abstractmethod
    def export(self, export_path, trial_id=None):
        pass

    @abstractmethod
    def to_taskflow(self, trial_id=None):
        pass

    def _override_arguments(self, config: Dict[str, Any], default_arguments: Any) -> Any:
        """
        Overrides the arguments with the provided hyperparameter config
        """
        new_arguments = copy.deepcopy(default_arguments)
        for key, value in config.items():
            if key.startswith(default_arguments.__class__.__name__):
                _, hp_key = key.split(".")
                setattr(new_arguments, hp_key, value)
        return new_arguments

    def _filter_model_candidates(self, language=None, preset=None) -> List[Dict[str, Any]]:
        """
        Model Candidates stored as Ray hyperparameter search space, organized by
        self.language and preset
        """
        model_candidates = self._model_candidates
        if language is not None:
            model_candidates = filter(lambda x: x["language"] == language, model_candidates)
        if preset is not None:
            model_candidates = filter(lambda x: x["preset"] == preset, model_candidates)
        hyperopt_search_space = {"candidates": hp.choice("candidates", list(model_candidates))}
        return hyperopt_search_space

    def _get_model_result(self, trial_id=None):
        if hasattr(self, "training_results"):
            if trial_id is not None:
                for result in self.training_results:
                    if result.metrics["trial_id"] == trial_id:
                        return result
                raise LookupError(
                    f"Trial_id '{trial_id}' is not found in 'training_results'. Did you enter the correct 'trial_id'?"
                )
            else:
                result = self.training_results.get_best_result(
                    metric=self.metric_for_best_model,
                    mode="max" if self.greater_is_better else "min",
                )
                return result
        else:
            raise AttributeError(
                "'AutoTrainer' has no attribute 'training_results'. Have you called the 'train' method?"
            )

    def show_training_results(self):
        if hasattr(self, "training_results"):
            return self.training_results.get_dataframe()
        else:
            raise AttributeError(
                "'AutoTrainer' has no attribute 'training_results'. Have you called the 'train' method?"
            )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_models: int = 1,
        preset: Optional[str] = None,
        num_gpus: Optional[int] = None,
        num_cpus: Optional[int] = None,
        max_concurrent_trials: Optional[int] = None,
        time_budget_s: Optional[Union[int, float, datetime.timedelta]] = None,
    ) -> ResultGrid:
        """
        Main logic of training models

        Args:
            train_dataset (Dataset, required): training dataset
            eval_dataset (Dataset, required): evaluation dataset
            num_models (int, required): number of model trials to run
            preset (str, optional): preset configuration for the trained models, can significantly impact accuracy, size, and inference latency of trained models. If not set, this will be inferred from data.
            num_gpus (str, optional): number of GPUs to use for the job. By default, this is set based on detected GPUs.
            num_cpus (str, optional): number of CPUs to use for the job. By default, this is set based on virtual cores.
            max_concurrent_trials (int, optional): maximum number of trials to run concurrently. Must be non-negative. If None or 0, no limit will be applied.
            time_budget_s: (int|float|datetime.timedelta, optional) global time budget in seconds after which all model trials are stopped.

        Returns:
            A set of objects for interacting with Ray Tune results. You can use it to inspect the trials and obtain the best result.
        """
        self._data_checks_and_inference(train_dataset, eval_dataset)
        trainable = self._construct_trainable(train_dataset, eval_dataset)
        model_search_space = self._filter_model_candidates(language=self.language, preset=preset)
        algo = HyperOptSearch(space=model_search_space, metric=self.metric_for_best_model, mode="max")
        algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent_trials)
        if num_gpus or num_cpus:
            hardware_resources = {}
            if num_gpus:
                hardware_resources["gpu"] = num_gpus
            if num_cpus:
                hardware_resources["cpu"] = num_cpus
            trainable = tune.with_resources(trainable, hardware_resources)
        tune_config = tune.tune_config.TuneConfig(num_samples=num_models, time_budget_s=time_budget_s, search_alg=algo)
        tuner = tune.Tuner(
            trainable,
            tune_config=tune_config,
            run_config=RunConfig(local_dir=self.output_dir) if self.output_dir else None,
        )
        self.training_results = tuner.fit()
        return self.training_results
