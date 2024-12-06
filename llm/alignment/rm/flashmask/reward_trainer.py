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

from collections import defaultdict

import paddle

from paddlenlp.trainer import Trainer


class RewardTrainer(Trainer):
    """
    Initialize RewardTrainer.
    """

    def __init__(self, model, data_collator, **kwargs):
        super().__init__(model, data_collator=data_collator, **kwargs)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if self.compute_metrics is not None:
            raise NotImplementedError("compute_metrics is not supported for RewardTrainer")

    def get_batch_metrics(self, model, batch, train_eval="train"):
        """Compute the RM loss and other metrics for the given batch of inputs for train or test."""
        rm_inputs = {
            "input_ids": batch["input_ids"],
            "position_ids": batch["position_ids"],
            "response_indexs": batch["response_indexs"],
        }

        if "attention_mask" in batch:
            rm_inputs["attention_mask"] = batch["attention_mask"]
        elif "attn_mask_start_row_indices" in batch:
            rm_inputs["attn_mask_start_row_indices"] = batch["attn_mask_start_row_indices"]
        elif "attn_mask_startend_row_indices" in batch:
            rm_inputs["attn_mask_startend_row_indices"] = batch["attn_mask_startend_row_indices"]

        loss, chosen_scores, rejected_scores = model(**rm_inputs)
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}
        metrics[f"{prefix}accuracy"] = (chosen_scores > rejected_scores).astype("float32").mean()
        for key in metrics:
            metrics[key] = self._nested_gather(paddle.tile(metrics[key], repeat_times=[1, 1])).mean().cpu()
        if self.args.should_save:
            self.store_metrics(metrics, train_eval=train_eval)
        return loss

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
            raise NotImplementedError("RewardTrainer only supports prediction_loss_only=True for now.")

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
