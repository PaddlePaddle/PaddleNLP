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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models import ScoreModelOutput
from paddle.io import Dataset

import paddlenlp.trainer.trainer as trainer
from paddlenlp.data import DataCollator
from paddlenlp.trainer import (
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from paddlenlp.trainer.utils import nested_detach
from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer

_tr_acc = None

speed_metrics = trainer.speed_metrics


def patch_speed_metrics(split, start_time, num_samples=None, num_steps=None, seq_length=None):
    # split: interval, train, eval, test
    result = speed_metrics(split, start_time, num_samples, num_steps, seq_length)
    if split not in ["train", "interval"]:
        return result
    # accuracy
    global _tr_acc
    tr_acc, total_acc_scalar, nested_gather = _tr_acc
    tr_acc_scalar = nested_gather(tr_acc).mean().item()
    total_acc_scalar += tr_acc_scalar
    tr_acc.subtract_(tr_acc)
    _tr_acc[1] = total_acc_scalar
    result["accuracy"] = round(tr_acc_scalar / num_steps, 8)
    if split == "train":
        result["train_accuracy"] = round(total_acc_scalar / num_steps, 8)
    return result


trainer.speed_metrics = patch_speed_metrics


def compute_accuracy(eval_pred) -> Dict[str, float]:
    higher_end_rewards, lower_end_rewards = eval_pred
    accuracy = (higher_end_rewards > lower_end_rewards).astype("float32").mean().item()
    rewards = np.concatenate([higher_end_rewards, lower_end_rewards], axis=0)
    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()
    return {
        "accuracy": accuracy,
        "rewards_mean": reward_mean,
        "rewards_std": reward_std,
    }


class RewardTrainer(Trainer):
    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
    ):
        if compute_metrics is None:
            compute_metrics = compute_accuracy

        super().__init__(
            model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        better_input_ids = inputs["better_input_ids"]
        worse_input_ids = inputs["worse_input_ids"]
        better_attention_mask = inputs["better_attention_mask"]
        worse_attention_mask = inputs["worse_attention_mask"]

        assert better_input_ids.shape[0] == worse_input_ids.shape[0], "batch size mismatch!"
        batch_size = better_input_ids.shape[0]

        output: ScoreModelOutput = model(
            paddle.concat([better_input_ids, worse_input_ids], axis=0),
            attention_mask=paddle.concat([better_attention_mask, worse_attention_mask], axis=0),
        )
        if isinstance(output, dict):
            scores = output.scores  # size = (2 * B, L, 1)
            end_scores = output.end_scores  # size = (2 * B, 1)
        else:
            scores, end_scores = output

        # size = (B, L)
        higher_rewards, lower_rewards = scores.squeeze(axis=-1).chunk(chunks=2, axis=0)
        # size = (B,)
        higher_end_rewards, lower_end_rewards = end_scores.squeeze(axis=-1).chunk(chunks=2, axis=0)

        if self.args.loss_type == "sequence-wise":
            loss = -F.log_sigmoid(higher_end_rewards - lower_end_rewards).mean()

            if self.args.regularization > 0.0:
                loss = loss + self.args.regularization * (
                    paddle.square(lower_end_rewards).mean() + paddle.square(higher_end_rewards).mean()
                )
        else:
            raise ValueError(f"Unknown loss type: {self.args.loss_type}")

        accuracy = (higher_end_rewards > lower_end_rewards).cast("float32").mean()  # size = ()

        # TODO(guosheng): use a formal way to replace this hack for accuracy track
        # in training
        global _tr_acc
        if _tr_acc is None:
            _tr_acc = [paddle.to_tensor(0.0), 0.0, self._nested_gather]
        _tr_acc[0] = _tr_acc[0] + accuracy.detach()

        if return_outputs:
            return loss, {
                "higher_end_rewards": higher_end_rewards,
                "lower_end_rewards": lower_end_rewards,
                "accuracy": accuracy,
            }
        return loss

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if self.args.pipeline_parallel_degree > 1:
            # hack for pipeline mode
            inputs = self._prepare_inputs(inputs)
            return self.prediction_pipeline_step(model, inputs, prediction_loss_only, ignore_keys)
        else:
            inputs = self._prepare_inputs(inputs)

        better_input_ids = inputs["better_input_ids"]
        worse_input_ids = inputs["worse_input_ids"]
        better_attention_mask = inputs["better_attention_mask"]
        worse_attention_mask = inputs["worse_attention_mask"]

        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                higher_rewards = self.model(better_input_ids, better_attention_mask)
                lower_rewards = self.model(worse_input_ids, worse_attention_mask)
            if isinstance(higher_rewards, dict):
                higher_end_rewards = higher_rewards.end_scores.squeeze(axis=-1)
                lower_end_rewards = lower_rewards.end_scores.squeeze(axis=-1)
            else:
                higher_end_rewards = higher_rewards[-1].squeeze(axis=-1)
                lower_end_rewards = lower_rewards[-1].squeeze(axis=-1)
        higher_end_rewards = nested_detach(higher_end_rewards)
        lower_end_rewards = nested_detach(lower_end_rewards)
        return None, higher_end_rewards.cast("float32"), lower_end_rewards.cast("float32")



class RMTrainer(Trainer):
    """
    Initialize RMTrainer.
    """

    def __init__(
        self,
        model,
        data_collator,
        **kwargs
    ):
        super().__init__(model, data_collator=data_collator, **kwargs)
        if self.compute_metrics is not None:
            raise NotImplementedError("compute_metrics is not supported for RMTrainer")


    def get_batch_metrics(self, model, batch, train_eval="train"):
        """Compute the RM loss and other metrics for the given batch of inputs for train or test."""
        rm_inputs = {
            "input_ids": batch["input_ids"],
            "position_ids": batch["position_ids"],
            "response_indexs": batch["response_indexs"]
        }
        if "attention_mask" in batch:
            rm_inputs["attention_mask"] = batch["attention_mask"]
        elif "attn_mask_start_row_indices" in batch:
            rm_inputs["attn_mask_start_row_indices"] = batch["attn_mask_start_row_indices"]
        elif "attn_mask_startend_row_indices" in batch:
            rm_inputs["attn_mask_startend_row_indices"] = batch["attn_mask_startend_row_indices"]

        loss = model(**rm_inputs)

        return loss
    
    def rm_criterion(self, scores, response_indexs):


    def compute_loss(self, model, inputs):
        """Compute the loss for the given batch of inputs."""
        loss = self.get_batch_metrics(model, inputs, train_eval="train")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """prediction_step"""
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                loss = self.get_batch_metrics(model, inputs, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)
        else:
            raise NotImplementedError("RMTrainer only supports prediction_loss_only=True for now.")
    
    def store_metrics(self, metrics, train_eval="train"):
        """store_metrics"""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs, **kwargs):
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = paddle.to_tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        if self.state.epoch is not None and train_eval == "train":
            self.state.epoch *= self.args.num_train_epochs
        return super().log(logs, **kwargs)
        