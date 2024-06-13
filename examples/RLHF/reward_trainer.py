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

        if self.args.loss_type == "token-wise":
            losses = []
            for i in range(batch_size):
                assert not paddle.all(
                    paddle.equal(better_input_ids[i], worse_input_ids[i]),
                ).item(), "The better and worse answers are the same!"
                higher_end_index = better_attention_mask[i].nonzero()[-1]
                lower_end_index = worse_attention_mask[i].nonzero()[-1]
                end_index = max(higher_end_index, lower_end_index)

                diverge_index = (better_input_ids[i] != worse_input_ids[i]).nonzero()[0]
                assert 0 <= diverge_index <= end_index, "diverge index is out of range!"

                # size = (L,)
                higher_truncated_rewards = higher_rewards[i, diverge_index : end_index + 1]
                lower_truncated_rewards = lower_rewards[i, diverge_index : end_index + 1]

                losses.append(
                    -F.log_sigmoid(higher_truncated_rewards - lower_truncated_rewards).mean(),
                )

                if self.args.regularization > 0.0:
                    losses[-1] = losses[-1] + self.args.regularization * (
                        paddle.square(lower_truncated_rewards).mean() + paddle.square(higher_truncated_rewards).mean()
                    )

            loss = paddle.stack(losses).mean()  # size = ()
        elif self.args.loss_type == "sequence-wise":
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
