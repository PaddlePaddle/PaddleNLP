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
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle.metric import Accuracy
from utils import DataArguments, ModelArguments, load_config, seq_convert_example

import paddlenlp

# from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import (
    ErnieForSequenceClassification,
    ErnieTokenizer,
    PretrainedTokenizer,
)
from paddlenlp.utils.log import logger


@dataclass
class DataCollatorWithDyanmicMaxLength:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        tokenizer (`paddlenlp.transformers.PretrainedTokenizer`):
            The tokenizer used for encoding the data.
    """

    tokenizer: PretrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pd"
    return_attention_mask: Optional[bool] = None
    label_list: Optional[int] = None

    # [32, 64, 128]
    dynamic_max_length: Optional[list[int]] = None

    def get_dynamic_max_length(self, examples):
        if not self.dynamic_max_length:
            return self.max_length

        if "sentence1" in examples[0]:
            lengths = list(
                map(
                    lambda example: len(
                        self.tokenizer.tokenize(example["sentence1"]) + self.tokenizer.tokenize(example["sentence2"])
                    )
                    + 3,
                    examples,
                )
            )

        else:
            lengths = list(map(lambda example: len(self.tokenizer.tokenize(example["sentence"])) + 2, examples))
        max_length = min(max(lengths), self.max_length)
        lengths = [length for length in self.dynamic_max_length if max_length < length]
        if not lengths:
            return self.max_length
        return lengths[0]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = self.get_dynamic_max_length(examples)
        examples = list(
            map(
                lambda example: seq_convert_example(
                    example,
                    tokenizer=self.tokenizer,
                    label_list=self.label_list,
                    max_seq_length=max_length,
                ),
                examples,
            )
        )
        features = {}
        for key in examples[0].keys():
            features[key] = np.array([example[key] for example in examples])

        if "label" in features:
            features["labels"] = features["label"]
            del features["label"]
        if "label_ids" in features:
            features["labels"] = features["label_ids"]
            del features["label_ids"]
        return features


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    model_args, data_args, training_args = load_config(
        model_args.config, "SequenceClassification", data_args.dataset, model_args, data_args, training_args
    )
    # Print model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    data_args.dataset = data_args.dataset.strip()
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.dataset)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    raw_datasets = load_dataset("clue", data_args.dataset)
    data_args.label_list = getattr(raw_datasets["train"], "label_list", None)
    num_classes = len(raw_datasets["train"].label_list)

    # Define tokenizer, model, loss function.
    model = ErnieForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)
    criterion = nn.loss.CrossEntropyLoss() if data_args.label_list else nn.loss.MSELoss()

    # Define data collector
    dynamic_max_length = None
    if data_args.dynamic_max_length:
        dynamic_max_length = [int(item) for item in data_args.dynamic_max_length.split(",")]

    data_collator = DataCollatorWithDyanmicMaxLength(
        tokenizer,
        max_length=data_args.max_seq_length,
        label_list=data_args.label_list,
        dynamic_max_length=dynamic_max_length,
    )

    # Dataset pre-process
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
    if training_args.do_eval:
        eval_dataset = raw_datasets["dev"]
    if training_args.do_predict:
        test_dataset = raw_datasets["test"]

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        metric = Accuracy()
        metric.reset()
        result = metric.compute(preds, label)
        metric.update(result)
        accu = metric.accumulate()
        metric.reset()
        return {"accuracy": accu}

    trainer = Trainer(
        model=model,
        criterion=criterion,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    if training_args.do_predict:
        test_ret = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_ret.metrics)
        logits = test_ret.predictions
        max_value = np.max(logits, axis=1, keepdims=True)
        exp_data = np.exp(logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {"label": probs.argmax(axis=-1).tolist(), "confidence": probs.max(axis=-1).tolist()}
        out_file = open(os.path.join(training_args.output_dir, "test_results.json"), "w")
        json.dump(out_dict, out_file)

    # Export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ]
        model_args.export_model_dir = os.path.join(model_args.export_model_dir, data_args.dataset, "export")
        paddlenlp.transformers.export_model(
            model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir
        )


if __name__ == "__main__":
    main()
