# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import random
from dataclasses import dataclass, field

import numpy as np
import paddle
from datasets import load_dataset
from paddle.metric import Accuracy

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    ErnieForSequenceClassification,
    ErnieTokenizer,
    LinearDecayWithWarmup,
)

METRIC_CLASSES = {
    "cola": Mcc,
    "sst2": Accuracy,
    "mrpc": AccuracyAndF1,
    "stsb": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
}


@dataclass
class ModelArguments:
    task_name: str = field(
        default=None,
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(METRIC_CLASSES.keys())},
    )
    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(
                sum([list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values()], [])
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_train_epoches: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    warmup_proportion: float = field(default=0.1, metadata={"help": "Linear warmup proportion over total steps."})


def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(seed)


def do_train():
    training_args, model_args = PdArgumentParser([TrainingArguments, ModelArguments]).parse_args_into_dataclasses()
    training_args: TrainingArguments = training_args
    model_args: ModelArguments = model_args

    training_args.print_config(model_args, "Model")
    training_args.print_config(training_args, "Training")

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    model_args.task_name = model_args.task_name.lower()

    sentence1_key, sentence2_key = task_to_keys[model_args.task_name]

    model_args.model_type = model_args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

    train_ds = load_dataset("glue", model_args.task_name, split="train")
    columns = train_ds.column_names
    is_regression = model_args.task_name == "stsb"
    label_list = None
    if not is_regression:
        label_list = train_ds.features["label"].names
        num_classes = len(label_list)
    else:
        num_classes = 1
    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, max_seq_len=model_args.max_seq_length)
        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=columns)
    data_collator = DataCollatorWithPadding(tokenizer)

    if model_args.task_name == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            "glue", model_args.task_name, split=["validation_matched", "validation_mismatched"]
        )
        dev_ds_matched = dev_ds_matched.map(preprocess_function, batched=True, remove_columns=columns)
        dev_ds_mismatched = dev_ds_mismatched.map(preprocess_function, batched=True, remove_columns=columns)
    else:
        dev_ds = load_dataset("glue", model_args.task_name, split="validation")
        dev_ds = dev_ds.map(preprocess_function, batched=True, remove_columns=columns)

    model = model_class.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = (
        training_args.max_steps
        if training_args.max_steps > 0
        else (len(train_ds) // training_args.per_device_train_batch_size * training_args.num_train_epochs)
    )
    warmup = training_args.warmup_steps if training_args.warmup_steps > 0 else model_args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(training_args.learning_rate, num_training_steps, warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=training_args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=training_args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    criterion = paddle.nn.loss.CrossEntropyLoss() if not is_regression else paddle.nn.loss.MSELoss()

    def compute_metrics(p):
        # Define the metrics of tasks.
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
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds
        if (training_args.do_eval and model_args.task_name != "mnli")
        else (dev_ds_matched if model_args.task_name == "mnli" else None),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=[optimizer, lr_scheduler],
    )

    # training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        if model_args.task_name != "mnli":
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        else:
            matched_eval_metrics = trainer.evaluate(eval_dataset=dev_ds_matched)
            trainer.log_metrics("matched_eval", matched_eval_metrics)
            trainer.save_metrics("matched_eval", matched_eval_metrics)
            mismatched_eval_metrics = trainer.evaluate(eval_dataset=dev_ds_mismatched)
            trainer.log_metrics("mismatched_eval", mismatched_eval_metrics)
            trainer.save_metrics("mismatched_eval", mismatched_eval_metrics)


if __name__ == "__main__":
    do_train()
