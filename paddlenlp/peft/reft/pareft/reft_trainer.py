# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import paddle

import paddlenlp.peft.reft.pavenv as pv
from paddlenlp.trainer import Trainer


@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""

    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, paddle.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs


class ReftTrainer(Trainer):
    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_intervention(save_directory=f"{output_dir}/intervenable_model", include_model=True)

    def _load_best_model(self):
        logging.warning(
            f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )
        self.model.load_intervention(f"{self.state.best_model_checkpoint}/intervenable_model", include_model=True)

    def compute_loss(self, intervenable: pv.IntervenableModel, inputs, return_outputs=False):
        # run intervened forward pass
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations={
                "sources->base": (
                    None,
                    # inputs["intervention_locations"].permute(1, 0, 2).tolist(),
                    inputs["intervention_locations"].transpose([1, 0, 2]).tolist(),
                )
            },
            labels=inputs["labels"],
        )
        return (cf_outputs[0], cf_outputs) if return_outputs else cf_outputs[0]
