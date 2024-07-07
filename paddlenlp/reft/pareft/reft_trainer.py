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

import os
from dataclasses import dataclass
from typing import Dict, Sequence

import paddle
from datasets import Dataset
from paddle.io import DataLoader
from tqdm import tqdm
from transformers.trainer_utils import EvalPrediction, denumpify_detensorize, has_length

# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging

import paddlenlp.reft.pavenv as pv
from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.trainer import Trainer

logger = logging.get_logger(__name__)


@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""

    # data_collator: DataCollator

    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, paddle.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(
    dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool
) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


class ReftTrainer(Trainer):
    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_intervention(save_directory=f"{output_dir}/intervenable_model", include_model=True)

    def _load_best_model(self):
        logger.warning(
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
            subspaces=(
                # inputs["subspaces"].permute(1, 0, 2).tolist()
                inputs["subspaces"].transpose([1, 0, 2]).tolist()
                if "subspaces" in inputs
                else None
            ),
        )
        # print("cf_outputs", cf_outputs)
        # return
        # return (cf_outputs.loss, cf_outputs) if return_outputs else cf_outputs.loss
        return (cf_outputs[0], cf_outputs) if return_outputs else cf_outputs[0]


class ReftTrainerForCausalLM(ReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        # return make_dataloader(
        #     self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True
        # )
        print(self.train_dataset)
        print("self.args", self.args)
        return make_dataloader(
            self.train_dataset,
            self.args.train_batch_size,
            self.data_collator,
            shuffle=True,
        )


class ReftTrainerForSequenceClassification(ReftTrainer):
    def evaluate(
        self,
        ignore_keys,
    ):

        # ensure everything is in eval mode
        self.model.model.eval()
        for k, v in self.model.interventions.items():
            _ = v[0].eval()

        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        eval_dataset = self.eval_dataset
        intervenable = self.model

        dataloader = make_dataloader(eval_dataset, batch_size, data_collator, shuffle=False)

        logger.info("***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        eval_iterator = tqdm(dataloader, position=0, leave=True)
        all_preds = []
        all_labels = []
        with paddle.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, paddle.Tensor):
                        inputs[k] = v.to(self.model.get_device())

                # [layers, batch_size, positions]
                intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).tolist()
                _, cf_outputs = intervenable(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                    unit_locations={"sources->base": (None, intervention_locations)},
                )

                all_preds += [cf_outputs.logits]
                all_labels += [inputs["labels"]]
        all_preds = paddle.concat(all_preds, dim=0).cpu().to(paddle.float32)
        all_labels = paddle.concat(all_labels, dim=0).cpu().to(paddle.float32)
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        metrics = denumpify_detensorize(metrics)

        metric_key_prefix = "eval"
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
