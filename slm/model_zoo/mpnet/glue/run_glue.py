# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import paddle
from paddle.io import DataLoader, Dataset
from paddle.metric import Accuracy

from paddlenlp.data import Pad, Stack
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments, set_seed
from paddlenlp.trainer.trainer_utils import speed_metrics
from paddlenlp.transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    ErnieForSequenceClassification,
    ErnieTokenizer,
    MPNetForSequenceClassification,
    MPNetTokenizer,
)


class MPNetTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        trans_func = partial(
            convert_example,
            tokenizer=self.tokenizer,
            label_list=self.args.label_list,
            max_seq_length=self.args.max_seq_length,
        )
        if self.args.task_name == "mnli":
            dev_ds_matched, dev_ds_mismatched = load_dataset(
                "glue", self.args.task_name, splits=["dev_matched", "dev_mismatched"]
            )

            dev_ds_matched = dev_ds_matched.map(trans_func, lazy=True)
            dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)
            dev_batch_sampler_matched = paddle.io.BatchSampler(
                dev_ds_matched, batch_size=self.args.per_device_eval_batch_size * 2, shuffle=False
            )
            dev_data_loader_matched = DataLoader(
                dataset=dev_ds_matched,
                batch_sampler=dev_batch_sampler_matched,
                collate_fn=self.data_collator,
                num_workers=2,
                return_list=True,
            )
            dev_batch_sampler_mismatched = paddle.io.BatchSampler(
                dev_ds_mismatched, batch_size=self.args.per_device_eval_batch_size * 2, shuffle=False
            )
            dev_data_loader_mismatched = DataLoader(
                dataset=dev_ds_mismatched,
                batch_sampler=dev_batch_sampler_mismatched,
                collate_fn=self.data_collator,
                num_workers=2,
                return_list=True,
            )
        else:
            dev_ds = load_dataset("glue", self.args.task_name, splits="dev")
            dev_ds = dev_ds.map(trans_func, lazy=True)
            dev_batch_sampler = paddle.io.BatchSampler(
                dev_ds, batch_size=self.args.per_device_eval_batch_size * 2, shuffle=False
            )
            dev_data_loader = DataLoader(
                dataset=dev_ds,
                batch_sampler=dev_batch_sampler,
                collate_fn=self.data_collator,
                num_workers=2,
                return_list=True,
            )

        start_time = time.time()

        if self.args.task_name == "mnli":
            output = self.evaluation_loop(
                dev_data_loader_matched,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.dataset_world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self.log(output.metrics)

            output = self.evaluation_loop(
                dev_data_loader_mismatched,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.dataset_world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self.log(output.metrics)

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

            self._memory_tracker.stop_and_update_metrics(output.metrics)

            return output.metrics
        else:
            output = self.evaluation_loop(
                dev_data_loader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.dataset_world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self.log(output.metrics)

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

            self._memory_tracker.stop_and_update_metrics(output.metrics)

            return output.metrics


METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "qnli": Accuracy,
    "mnli": Accuracy,
    "rte": Accuracy,
    "wnli": Accuracy,
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "electra": (ElectraForSequenceClassification, ElectraTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
    "mpnet": (MPNetForSequenceClassification, MPNetTokenizer),
}


@dataclass
class ModelArguments:
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The name of the task to train selected in the list: " + ", ".join(METRIC_CLASSES.keys()))},
    )
    model_type: Optional[str] = field(
        default="convbert",
        metadata={"help": ("Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))},
    )
    model_name_or_path: Optional[str] = field(
        default="convbert-base",
        metadata={
            "help": (
                "Path to pre-trained model or shortcut name selected in the list: "
                + ", ".join(
                    sum(
                        [list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values()],
                        [],
                    )
                ),
            )
        },
    )
    layer_lr_decay: Optional[float] = field(
        default=1.0,
        metadata={"help": ("layer_lr_decay")},
    )


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default="./data",
        metadata={"help": "The path of datasets to be loaded."},
    )


def compute_metrics(eval_preds, metric):
    labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
    preds = paddle.to_tensor(eval_preds.predictions)
    preds = paddle.nn.functional.softmax(preds, axis=-1)
    labels = paddle.argmax(labels, axis=-1)
    correct = metric.compute(preds, labels)
    metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        return {
            "acc": res[0],
            "precision": res[1],
            "recall": res[2],
            "f1": res[3],
            "acc and f1": res[4],
        }
    elif isinstance(metric, Mcc):
        return {
            "mcc": res[0],
        }
    elif isinstance(metric, PearsonAndSpearman):
        return {
            "pearson": res[0],
            "spearman": res[1],
            "pearson and spearman": res[2],
        }
    else:
        return {
            "acc": res,
        }


def convert_example(example, tokenizer, label_list, max_seq_length=512, is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example["sentence"], max_length=max_seq_length, return_token_type_ids=True)
    else:
        example = tokenizer(
            example["sentence1"],
            text_pair=example["sentence2"],
            max_length=max_seq_length,
            return_token_type_ids=True,
        )

    if not is_test:
        return example["input_ids"], example["token_type_ids"], label
    else:
        return example["input_ids"], example["token_type_ids"]


@dataclass
class DataCollator:
    def __init__(self, tokenizer, train_ds):
        self.tokenizer = (tokenizer,)
        self.train_ds = (train_ds,)

    def __call__(self, features):
        input_ids = []
        labels = []
        batch = {}

        for feature in features:
            input_idx, _, label = feature
            input_ids.append(input_idx)
            labels.append(label)

        if not isinstance(self.tokenizer, MPNetTokenizer):
            self.tokenizer = self.tokenizer[0]
            self.train_ds = self.train_ds[0]
        input_ids = (Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(input_ids),)  # input_ids
        labels = (Stack(dtype="int64" if self.train_ds.label_list else "float32")(labels),)  # labels

        batch["input_ids"] = input_ids[0]
        batch["labels"] = labels[0]

        return batch


def _get_layer_lr_radios(layer_decay=0.8, n_layers=12):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = OrderedDict(
        {
            "mpnet.embeddings.": 0,
            "mpnet.encoder.relative_attention_bias.": 0,
            "mpnet.pooler.": n_layers + 2,
            "mpnet.classifier.": n_layers + 2,
        }
    )
    for layer in range(n_layers):
        key_to_depths[f"mpnet.encoder.layer.{str(layer)}."] = layer + 1
    return {key: (layer_decay ** (n_layers + 2 - depth)) for key, depth in key_to_depths.items()}


def do_train():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.output_dir is None:
        training_args.output_dir = model_args.task_name.lower()
    if model_args.task_name is not None:
        training_args.task_name = model_args.task_name
    if model_args.max_seq_length is not None:
        training_args.max_seq_length = model_args.max_seq_length

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    model_args.task_name = model_args.task_name.lower()
    metric_class = METRIC_CLASSES[model_args.task_name]
    model_args.model_type = model_args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

    train_ds = load_dataset("glue", model_args.task_name, splits="train")
    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)
    training_args.label_list = train_ds.label_list

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=training_args.label_list,
        max_seq_length=model_args.max_seq_length,
    )
    train_ds = train_ds.map(trans_func, lazy=True)
    batchify_fn = DataCollator(tokenizer, train_ds)

    num_classes = 1 if training_args.label_list is None else len(training_args.label_list)
    model = model_class.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)

    if model_args.layer_lr_decay != 1.0:
        layer_lr_radios_map = _get_layer_lr_radios(model_args.layer_lr_decay, n_layers=12)
        for name, parameter in model.named_parameters():
            layer_lr_radio = 1.0
            for k, radio in layer_lr_radios_map.items():
                if k in name:
                    layer_lr_radio = radio
                    break
            parameter.optimize_attr["learning_rate"] *= layer_lr_radio

    loss_fct = paddle.nn.loss.CrossEntropyLoss() if training_args.label_list else paddle.nn.loss.MSELoss()

    metric = metric_class()
    compute_metrics_func = partial(
        compute_metrics,
        metric=metric,
    )

    trainer = MPNetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=batchify_fn,
        criterion=loss_fct,
        compute_metrics=compute_metrics_func,
    )

    if training_args.do_train:
        train_results = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_results.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)


if __name__ == "__main__":
    do_train()
