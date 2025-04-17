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

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import paddle
from paddle.metric import Accuracy

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    export_model,
)
from paddlenlp.utils.log import logger

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    task_name: str = field(
        default=None,
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(METRIC_CLASSES.keys())},
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        }
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the exported inference model."},
    )
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_rank: int = field(default=8, metadata={"help": "Lora rank"})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha"})
    qat: bool = field(default=False, metadata={"help": "Whether to use QAT technique"})
    qat_type: str = field(default="A8W8", metadata={"help": "Quantization type. Supported values: A8W8, W4,A8W4"})


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
        example = tokenizer(example["sentence"], max_seq_len=max_seq_length)
    else:
        example = tokenizer(example["sentence1"], text_pair=example["sentence2"], max_seq_len=max_seq_length)

    if not is_test:
        return {"input_ids": example["input_ids"], "token_type_ids": example["token_type_ids"], "labels": label}
    else:
        return {"input_ids": example["input_ids"], "token_type_ids": example["token_type_ids"]}


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

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

    data_args.task_name = data_args.task_name.strip().lower()
    metric = METRIC_CLASSES[data_args.task_name]()

    train_ds = load_dataset("glue", data_args.task_name, splits="train")
    if model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    trans_func = partial(
        convert_example, tokenizer=tokenizer, label_list=train_ds.label_list, max_seq_length=data_args.max_seq_length
    )
    train_ds = train_ds.map(trans_func, lazy=True)

    if data_args.task_name == "mnli":
        dev_ds, dev_ds_mismatched = load_dataset("glue", data_args.task_name, splits=["dev_matched", "dev_mismatched"])

        dev_ds = dev_ds.map(trans_func, lazy=True)
        dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)

        test_ds, test_ds_mismatched = load_dataset(
            "glue", data_args.task_name, splits=["test_matched", "test_mismatched"]
        )

        test_ds = test_ds.map(trans_func, lazy=True)
        test_ds_mismatched = test_ds_mismatched.map(trans_func, lazy=True)

    else:
        dev_ds = load_dataset("glue", data_args.task_name, splits="dev")
        dev_ds = dev_ds.map(trans_func, lazy=True)

        test_ds = load_dataset("glue", data_args.task_name, splits="test")
        test_ds = test_ds.map(trans_func, lazy=True)

    # Define data collector
    data_collator = DataCollatorWithPadding(tokenizer)
    num_classes = 1 if train_ds.label_list is None else len(train_ds.label_list)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)
    dtype = "float32"
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"
    if model_args.lora:
        # TODO: hardcode parameters for now. Change after MergedLoRA is introduced
        lora_config = LoRAConfig(
            target_modules=[
                ".*self_attn.q_proj.*",
                ".*self_attn.k_proj.*",
                ".*self_attn.v_proj.*",
                ".*self_attn.out_proj.*",
                ".*linear1.*",
                ".*linear2.*",
            ],
            trainable_modules=[".*classifier.*"],
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            merge_weights=False,
            dtype=dtype,
        )
        model = LoRAModel(model, lora_config)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    if model_args.qat:
        from paddle import nn
        from paddle.quantization import QAT, QuantConfig
        from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
        from paddleslim.quant.quanters import FakeQuanterChannelWiseAbsMaxObserver

        from paddlenlp.peft.lora import LoRALinear
        from paddlenlp.peft.lora.lora_quant_layers import QuantedLoRALinear

        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)

        if model_args.qat_type == "A8W8":
            activation = FakeQuanterWithAbsMaxObserver(moving_rate=0.9, bit_length=8, dtype=dtype)
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=8, dtype=dtype)
        elif model_args.qat_type == "W4":
            activation = None
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype=dtype)
        elif model_args.qat_type == "A8W4":
            activation = FakeQuanterWithAbsMaxObserver(moving_rate=0.9, bit_length=8, dtype=dtype)
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype=dtype)
        else:
            raise ValueError("qat_type should be one of ['A8W8', 'W4', 'A8W4']")

        q_config.add_type_config(LoRALinear, weight=weight, activation=activation)
        q_config.add_type_config(nn.Linear, weight=weight, activation=activation)

        qat = QAT(q_config)
        model = qat.quantize(model, inplace=True)

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        metric.reset()
        result = metric.compute(preds, label)
        metric.update(result)
        res = metric.accumulate()
        metric.reset()
        if isinstance(metric, AccuracyAndF1):
            return {
                "accuracy": res[0],
                "precision": res[1],
                "recall": res[2],
                "f1 score": res[3],
                "accuracy and f1": res[4],
            }
        elif isinstance(metric, Mcc):
            return {"mcc": res[0]}
        elif isinstance(metric, PearsonAndSpearman):
            return {
                "pearson": res[0],
                "spearman": res[1],
                "pearson and spearman": res[2],
            }
        else:
            return {"accuracy": res}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds if training_args.do_eval else None,
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
        logger.info("*** Evaluate ***")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        if data_args.task_name == "mnli":
            eval_metrics = trainer.evaluate(dev_ds_mismatched)
            trainer.log_metrics("eval", eval_metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        test_ret = trainer.predict(test_ds)
        trainer.log_metrics("test", test_ret.metrics)
        if data_args.task_name == "mnli":
            test_ret = trainer.predict(test_ds_mismatched)
            trainer.log_metrics("test", test_ret.metrics)

    # export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir, "export")
        export_model(model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir)


if __name__ == "__main__":
    main()
