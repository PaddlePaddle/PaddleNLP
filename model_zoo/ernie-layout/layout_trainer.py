# encoding=utf-8
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

import json
import os
from typing import Dict

from paddlenlp.trainer import Trainer


class LayoutTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, convert_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.convert_fn = convert_fn

    def save_predictions(self, split, preds, labels):
        """
        Save metrics into a json file for that split, e.g. `train_results.json`.
        Under distributed environment this is done only for a process with rank 0.
        Args:
            split (`str`):
                Mode/split name: one of `train`, `eval`, `test`, `all`
        To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
        unformatted numbers are saved in the current method.
        """

        path = os.path.join(self.args.output_dir, f"{split}_predictions.json")
        with open(path, "w") as f:
            json.dump(preds, f, ensure_ascii=False, indent=4, sort_keys=True)

        path = os.path.join(self.args.output_dir, f"{split}_golden_labels.json")
        with open(path, "w") as f:
            json.dump(labels, f, ensure_ascii=False, indent=4, sort_keys=True)

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ) -> Dict[str, float]:

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            pred_rst, gt_rst, eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, output.label_ids
            )
            self.save_predictions("eval", pred_rst, gt_rst)
            metrics = self.compute_metrics(eval_preds)
            if self.convert_fn is not None:
                processed_metrics = self.convert_fn(pred_rst, self.args.output_dir)
                if processed_metrics is not None:
                    metrics.update(processed_metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):

        predict_dataloader = self.get_test_dataloader(predict_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            pred_rst, gt_rst, eval_preds = self.post_process_function(
                predict_examples, predict_dataset, output.predictions, output.label_ids
            )
            self.save_predictions("test", pred_rst, gt_rst)
            metrics = self.compute_metrics(eval_preds)

            if self.convert_fn is not None:
                processed_metrics = self.convert_fn(pred_rst, self.args.output_dir)
                if processed_metrics is not None:
                    metrics.update(processed_metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        else:
            metrics = {}
        return metrics
