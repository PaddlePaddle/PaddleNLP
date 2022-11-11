from abc import ABCMeta, abstractmethod
from pickle import NONE
from ray import tune
from typing import Callable, Dict, Any
import copy
from paddle.io import Dataset

from paddlenlp.trainer import TrainingArguments, CompressionArguments


class AutoTrainerBase(metaclass=ABCMeta):
    
    def __init__(self, metric_for_best_model=None, preset=None, language="Chinese", **kwargs):
        self.metric_for_best_model = metric_for_best_model
        self.preset = preset
        self.language = language

    @property
    @abstractmethod
    def _default_training_argument(self) -> TrainingArguments:
        pass

    @property
    @abstractmethod
    def _default_compress_argument(self) -> CompressionArguments:
        pass

    @property
    @abstractmethod
    def _model_candidates(self) -> Dict[str, Any]: 
        pass
    
    @abstractmethod
    def _data_checks_and_inference(self, train_dataset, eval_dataset):
        pass

    @abstractmethod
    def _construct_trainable(self, train_dataset, eval_dataset) -> Callable:
        pass
    
    @abstractmethod
    def _compute_metrics(self, eval_preds) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def _preprocess_fn(self, dataset, tokenizer, max_seq_length, is_test=False) -> Dataset:
        pass

    def _override_arguments(self, config, default_arguments) -> Any:
        new_arguments = copy.deepcopy(default_arguments)
        for key, value in config.items():
            if key.startswith(default_arguments.__class__.__name__):
                _, hp_key = key.split(".")
                setattr(new_arguments, hp_key, value)
        return new_arguments
    
    def _override_training_arguments(self, config) -> TrainingArguments:
        return self._override_arguments(config, self._default_training_argument)

    def _override_compression_arguments(self, config) -> CompressionArguments:
        return self._override_arguments(config, self._default_compress_argument)
        
    def train(self,
        train_dataset,
        eval_dataset,
        num_models=1,
        num_gpus=None,
        num_cpus=None,
        max_concurrent_trials=None,
        time_budget_s=None
        ):
        self._data_checks_and_inference(train_dataset, eval_dataset)
        trainable = self._construct_trainable(train_dataset, eval_dataset)
        if num_gpus or num_cpus:
            hardware_resources = {}
            if num_gpus:
                hardware_resources["gpu"] = num_gpus
            if num_cpus:
                hardware_resources["cpu"] = num_cpus
            trainable = tune.with_resources(trainable, hardware_resources)
        tune_config = tune.tune_config.TuneConfig(
            num_samples=num_models,
            time_budget_s=time_budget_s,
            max_concurrent_trials=max_concurrent_trials)
        tuner = tune.Tuner(
            trainable,
            param_space=self._model_candidates[self.preset],
            tune_config=tune_config)
        self.training_results = tuner.fit()
        return self.training_results
