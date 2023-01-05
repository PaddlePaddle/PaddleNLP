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
import os
import shutil
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
from paddlenlp.utils.log import logger


class AutoTrainerBase(metaclass=ABCMeta):
    """
    The meta classs of AutoTrainer, which contains the common properies and methods of AutoNLP.
    Task-specific AutoTrainers need to inherit from the meta class.

    Args:
        train_dataset (Dataset, required): Training dataset, must contains the 'text_column' and 'label_column' specified below
        eval_dataset (Dataset, required): Evaluation dataset, must contains the 'text_column' and 'label_column' specified below
        language (string, required): language of the text
        metric_for_best_model (string, optional): the name of the metrc for selecting the best model.
        greater_is_better (bool, required): Whether better models should have a greater metric or not. Use in conjuction with `metric_for_best_model`.
        output_dir (str, optional): Output directory for the experiments, defaults to "autpnlp_results"
    """

    training_path = "training"
    export_path = "exported_model"
    results_filename = "experiment_results.csv"

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
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
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
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

    def export(self, export_path, trial_id=None):
        """
        Export the model from a certain `trial_id` to the given file path.

        Args:
            export_path (str, required): the filepath to export to
            trial_id (int, required): use the `trial_id` to select the model to export. Defaults to the best model selected by `metric_for_best_model`
        """
        model_result = self._get_model_result(trial_id=trial_id)
        exported_model_path = os.path.join(model_result.log_dir, self.export_path)
        shutil.copytree(exported_model_path, export_path)
        logger.info(f"Exported to {export_path}")

    @abstractmethod
    def to_taskflow(self, trial_id=None):
        """
        Convert the model from a certain `trial_id` to a Taskflow for model inference

        Args:
            trial_id (int, required): use the `trial_id` to select the model to export. Defaults to the best model selected by `metric_for_best_model`
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, trial_id=None, eval_dataset=None) -> Dict[str, float]:
        """
        Evaluate the models from a certain `trial_id` on the given dataset

        Args:
            trial_id (str, optional): specify the model to be evaluated through the `trial_id`. Defaults to the best model selected by `metric_for_best_model`
            eval_dataset (Dataset, optional): custom evaluation dataset and must contains the 'text_column' and 'label_column' fields.
                If not provided, defaults to the evaluation dataset used at construction
        """
        raise NotImplementedError

    def _override_hp(self, config: Dict[str, Any], default_hp: Any) -> Any:
        """
        Overrides the arguments with the provided hyperparameter config
        """
        new_hp = copy.deepcopy(default_hp)
        for key, value in config.items():
            if key.startswith(default_hp.__class__.__name__):
                _, hp_key = key.split(".")
                setattr(new_hp, hp_key, value)
        return new_hp

    def _filter_model_candidates(
        self, language=None, preset=None, custom_model_candidates=None
    ) -> List[Dict[str, Any]]:
        """
        Model Candidates stored as Ray hyperparameter search space, organized by
        override, language and preset
        """
        model_candidates = custom_model_candidates if custom_model_candidates is not None else self._model_candidates
        if language is not None:
            model_candidates = filter(
                lambda x: x["language"] == language if "language" in x else True, model_candidates
            )
        if preset is not None:
            model_candidates = filter(lambda x: x["preset"] == preset if "preset" in x else True, model_candidates)
        return list(model_candidates)

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

    def set_log_level(self):
        if self.verbosity > 0:
            logger.set_level("WARNING")
        else:
            logger.set_level("INFO")

    def show_training_results(self):
        if hasattr(self, "training_results"):
            return self.training_results.get_dataframe()
        else:
            raise AttributeError(
                "'AutoTrainer' has no attribute 'training_results'. Have you called the 'train' method?"
            )

    def load(self, path: str):
        """
        Restores the AutoTrainer from a given experiment directory produced by a previous run

        Args:
            path (str, required): The filepath to load the previous experiments
        """
        logger.info(f"Restoring from {path}")
        self.training_results = self.tuner.get_results()
        self.tuner = tune.Tuner.restore(path)
        logger.info("Found existing training results.")

    def train(
        self,
        num_models: int = 1,
        preset: Optional[str] = None,
        num_gpus: Optional[int] = None,
        num_cpus: Optional[int] = None,
        max_concurrent_trials: Optional[int] = None,
        time_budget_s: Optional[Union[int, float, datetime.timedelta]] = None,
        experiment_name: str = None,
        verbosity: int = 0,
        hp_overrides: Dict[str, Any] = None,
        custom_model_candidates: List[Dict[str, Any]] = None,
    ) -> ResultGrid:
        """
        Main logic of training models

        Args:
            num_models (int, required): number of model trials to run
            preset (str, optional): preset configuration for the trained models, can significantly impact accuracy, size, and inference latency of trained models.
                If not set, this will be inferred from data.
            num_gpus (str, optional): number of GPUs to use for the job. By default, this is set based on detected GPUs.
            num_cpus (str, optional): number of CPUs to use for the job. By default, this is set based on virtual cores.
            max_concurrent_trials (int, optional): maximum number of trials to run concurrently. Must be non-negative. If None or 0, no limit will be applied.
            time_budget_s: (int|float|datetime.timedelta, optional) global time budget in seconds after which all model trials are stopped.
            experiment_name: (str, optional): name of the experiment. Experiment log will be stored under <output_dir>/<experiment_name>.
                Defaults to UNIX timestamp.
            verbosity: (int, optional): controls the verbosity of the logger. Defaults to 0, which set the logger level at INFO. To reduce the amount of logs,
                use verbosity > 0 to set the logger level to WARNINGS
            hp_overrides: (dict[str, Any], optional): Advanced users only.
                override the hyperparameters of every model candidate.  For example, {"TrainingArguments.max_steps": 5}.
            custom_model_candiates: (dict[str, Any], optional): Advanced users only.
                Run the user-provided model candidates instead of the default model candidated from PaddleNLP. See `._model_candidates` property as an example

        Returns:
            A set of objects for interacting with Ray Tune results. You can use it to inspect the trials and obtain the best result.
        """
        # Changing logger verbosity here doesn't work. Need to change in the worker's code via the _construct_trainable method.
        self.verbosity = verbosity

        if hasattr(self, "tuner") and self.tuner is not None:
            logger.info("Overwriting the existing Tuner and any previous training results")

        self._data_checks_and_inference()
        trainable = self._construct_trainable()
        model_candidates = self._filter_model_candidates(
            language=self.language, preset=preset, custom_model_candidates=custom_model_candidates
        )
        if hp_overrides is not None:
            for model_candidate in model_candidates:
                model_candidate.update(hp_overrides)
        search_space = {"candidates": hp.choice("candidates", model_candidates)}
        algo = HyperOptSearch(space=search_space, metric=self.metric_for_best_model, mode="max")
        algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent_trials)
        if num_gpus or num_cpus:
            hardware_resources = {}
            if num_gpus:
                hardware_resources["gpu"] = num_gpus
            if num_cpus:
                hardware_resources["cpu"] = num_cpus
            trainable = tune.with_resources(trainable, hardware_resources)
        tune_config = tune.tune_config.TuneConfig(num_samples=num_models, time_budget_s=time_budget_s, search_alg=algo)

        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%s")

        self.tuner = tune.Tuner(
            trainable,
            tune_config=tune_config,
            run_config=RunConfig(
                name=experiment_name, log_to_file=True, local_dir=self.output_dir if self.output_dir else None
            ),
        )
        self.training_results = self.tuner.fit()
        self.show_training_results().to_csv(
            path_or_buf=os.path.join(self.output_dir, experiment_name, self.results_filename), index=False
        )
        return self.training_results
