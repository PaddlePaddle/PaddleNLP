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
import logging
import os
import shutil
import sys
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import ray
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
        verbosity: (int, optional): controls the verbosity of the run. Defaults to 1, which let the workers log to the driver.To reduce the amount of logs,
                use verbosity > 0 to set stop the workers from logging to the driver.
    """

    training_path = "training_checkpoints"  # filepath for Trainer's training checkpoints
    save_path = "trained_model"  # filepath for the trained dygraph model
    export_path = "exported_model"  # filepath for the exported static model
    compress_path = "compressed_model"  # filepath for the compressed static model
    results_filename = "experiment_results.csv"  # filepath for storing experiment results
    experiment_path = None  # filepath for the experiment results
    visualdl_path = "visualdl"  # filepath for the visualdl

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        metric_for_best_model: str,
        greater_is_better: bool,
        language: str = "Chinese",
        output_dir: str = "autonlp_results",
        verbosity: int = 1,
        **kwargs,
    ):
        if metric_for_best_model is not None and not metric_for_best_model.startswith("eval_"):
            self.metric_for_best_model = f"eval_{metric_for_best_model}"
        else:
            self.metric_for_best_model = metric_for_best_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.greater_is_better = greater_is_better
        if language not in self.supported_languages:
            raise ValueError(
                f"'{language}' is not supported. Please choose among the following: {self.supported_languages}"
            )

        self.language = language
        self.output_dir = output_dir
        self.kwargs = kwargs
        # Per default, Ray Tune creates JSON, CSV and TensorBoardX logger callbacks, turning it off
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
        # use log_to_driver to control verbosity
        ray.init(ignore_reinit_error=True, log_to_driver=True if verbosity >= 1 else False)

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """
        Override to store the supported languages for each auto trainer class
        """

    @property
    @abstractmethod
    def _default_training_argument(self) -> TrainingArguments:
        """
        Default TrainingArguments for the Trainer
        """
        return TrainingArguments(
            output_dir=self.training_path,
            disable_tqdm=True,
            load_best_model_at_end=True,
            save_total_limit=1,
            report_to=["visualdl", "autonlp"],
            logging_dir=self.visualdl_path,  # if logging_dir is redefined, the function visualdl() should be redefined as well.
        )

    @property
    @abstractmethod
    def _model_candidates(self) -> List[Dict[str, Any]]:
        """
        Model Candidates stored as Ray hyperparameter search space, organized by
        self.language and preset
        """

    @abstractmethod
    def _data_checks_and_inference(self, dataset_list: List[Dataset]):
        """
        Performs different data checks and inferences on the datasets
        """

    def _construct_trainable(self) -> Callable:
        """
        Returns the Trainable functions that contains the main preprocessing and training logic
        """

        def trainable(model_config):
            # import is required for proper pickling
            from paddlenlp.utils.log import logger

            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(logger.format)
            logger.logger.addHandler(stdout_handler)

            # construct trainer
            model_config = model_config["candidates"]
            trainer = self._construct_trainer(model_config)
            # train
            trainer.train()
            # evaluate
            eval_metrics = trainer.evaluate()
            # save dygraph model
            trainer.save_model(self.save_path)

            if os.path.exists(self.training_path):
                logger.info("Removing training checkpoints to conserve disk space")
                shutil.rmtree(self.training_path)
            return eval_metrics

        return trainable

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
    def export(self, export_path: str, trial_id: Optional[str] = None):
        """
        Export the model from a certain `trial_id` to the given file path.

        Args:
            export_path (str, required): the filepath to export to
            trial_id (int, optional): use the `trial_id` to select the model to export. Defaults to the best model selected by `metric_for_best_model`
        """

        raise NotImplementedError

    @abstractmethod
    def to_taskflow(self, trial_id: Optional[str] = None):
        """
        Convert the model from a certain `trial_id` to a Taskflow for model inference

        Args:
            trial_id (int, optional): use the `trial_id` to select the model to export. Defaults to the best model selected by `metric_for_best_model`
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, eval_dataset: Optional[Dataset] = None, trial_id: Optional[str] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics from a certain `trial_id` on the given dataset.

        Args:
            trial_id (str, optional): specify the model to be evaluated through the `trial_id`. Defaults to the best model selected by `metric_for_best_model`
            eval_dataset (Dataset, optional): custom evaluation dataset and must contains the 'text_column' and 'label_column' fields.
                If not provided, defaults to the evaluation dataset used at construction.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_dataset: Dataset, trial_id: Optional[str] = None):
        """
        Run prediction and returns predictions and potential metrics from a certain `trial_id` on the given dataset
        Args:
            test_dataset (Dataset, required): Custom test dataset and must contains the 'text_column' and 'label_column' fields.
            trial_id (str, optional): Specify the model to be evaluated through the `trial_id`. Defaults to the best model selected by `metric_for_best_model`.
        """
        raise NotImplementedError

    def _override_hp(self, config: Dict[str, Any], default_hp: Any) -> Any:
        """
        Overrides the arguments with the provided hyperparameter config
        """
        new_hp = copy.deepcopy(default_hp)
        for key, value in config.items():
            if key in new_hp.to_dict():
                if key in ["output_dir", "logging_dir"]:
                    logger.warning(f"{key} cannot be overridden")
                else:
                    setattr(new_hp, key, value)
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
        self.tuner = tune.Tuner.restore(path)
        self.training_results = self.tuner.get_results()
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
            hp_overrides: (dict[str, Any], optional): Advanced users only.
                override the hyperparameters of every model candidate.  For example, {"max_steps": 5}.
            custom_model_candiates: (dict[str, Any], optional): Advanced users only.
                Run the user-provided model candidates instead of the default model candidated from PaddleNLP. See `._model_candidates` property as an example

        Returns:
            A set of objects for interacting with Ray Tune results. You can use it to inspect the trials and obtain the best result.
        """
        if hasattr(self, "tuner") and self.tuner is not None:
            logger.info("Overwriting the existing Tuner and any previous training results")

        trainable = self._construct_trainable()
        model_candidates = self._filter_model_candidates(
            language=self.language, preset=preset, custom_model_candidates=custom_model_candidates
        )
        if hp_overrides is not None:
            for model_candidate in model_candidates:
                model_candidate.update(hp_overrides)
        search_space = {"candidates": hp.choice("candidates", model_candidates)}
        mode = "max" if self.greater_is_better else "min"
        algo = HyperOptSearch(space=search_space, metric=self.metric_for_best_model, mode=mode)
        algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent_trials)
        if num_gpus or num_cpus:
            hardware_resources = {}
            if num_gpus:
                hardware_resources["gpu"] = num_gpus
            if num_cpus:
                hardware_resources["cpu"] = num_cpus
            trainable = tune.with_resources(trainable, hardware_resources)

        def trial_creator(trial):
            return "{}".format(trial.trial_id)

        tune_config = tune.TuneConfig(
            num_samples=num_models,
            time_budget_s=time_budget_s,
            search_alg=algo,
            trial_name_creator=trial_creator,
            trial_dirname_creator=trial_creator,
        )

        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%s")
        self.experiment_path = os.path.join(self.output_dir, experiment_name)

        self.tuner = tune.Tuner(
            trainable,
            tune_config=tune_config,
            run_config=RunConfig(
                name=experiment_name,
                log_to_file="train.log",
                local_dir=self.output_dir if self.output_dir else None,
                callbacks=[tune.logger.CSVLoggerCallback()],
            ),
        )
        self.training_results = self.tuner.fit()
        self.show_training_results().to_csv(
            path_or_buf=os.path.join(self.output_dir, experiment_name, self.results_filename), index=False
        )

        return self.training_results

    def visualdl(self, trial_id: Optional[str] = None):
        """
        Return visualdl path to represent the results of the taskflow training.
        """
        model_result = self._get_model_result(trial_id=trial_id)
        return os.path.join(model_result.log_dir, self.visualdl_path)
