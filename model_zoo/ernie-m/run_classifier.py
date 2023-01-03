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


import random
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from datasets import load_dataset
from paddle.io import Dataset
from paddle.metric import Accuracy
from paddle.optimizer import AdamW

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.ops.optimizer import layerwise_lr_decay
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import (
    AutoTokenizer,
    ErnieMForSequenceClassification,
    LinearDecayWithWarmup,
)

all_languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
task_type_list = ["cross-lingual-transfer", "translate-train-all"]


@dataclass
class ModelArguments:
    task_type: str = field(
        default=None,
        metadata={"help": "The type of the task to finetune selected in the list: " + ", ".join(task_type_list)},
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(list(ErnieMForSequenceClassification.pretrained_init_configuration.keys()))
        },
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    classifier_dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    layerwise_decay: float = field(default=0.8, metadata={"help": "Layerwise decay ratio."})


def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(seed + paddle.distributed.get_rank())`
    paddle.seed(seed)


def convert_example(example, tokenizer, max_seq_length=256):
    """convert a example into necessary features"""
    # Convert raw text to feature
    tokenized_example = tokenizer(
        example["premise"],
        text_pair=example["hypothesis"],
        max_length=max_seq_length,
        padding=False,
        truncation=True,
        return_position_ids=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    return tokenized_example


class XnliDataset(Dataset):
    """
    Make all languages datasets be loaded in lazy mode.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        # Ar language has 2000 empty data.
        self.num_samples = [len(i) for i in datasets]
        self.cumsum_len = np.cumsum(self.num_samples)

    def __getitem__(self, idx):
        language_idx = np.argmax(self.cumsum_len > idx)
        last = language_idx - 1 if language_idx > 0 else language_idx
        sample_idx = idx - self.cumsum_len[last] if idx >= self.cumsum_len[last] else idx
        return self.datasets[int(language_idx)][int(sample_idx)]

    def __len__(self):
        return self.cumsum_len[-1]


def do_train():
    training_args, model_args = PdArgumentParser([TrainingArguments, ModelArguments]).parse_args_into_dataclasses()
    training_args: TrainingArguments = training_args
    model_args: ModelArguments = model_args

    training_args.print_config(model_args, "Model")

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=model_args.max_seq_length)
    remove_columns = ["premise", "hypothesis"]
    if model_args.task_type == "cross-lingual-transfer":
        train_ds = load_dataset("xnli", "en", split="train")
        eval_ds = load_dataset("xnli", "en", split="test")
        train_ds = train_ds.map(trans_func, batched=True, remove_columns=remove_columns)
        eval_ds = eval_ds.map(trans_func, batched=True, remove_columns=remove_columns)
    elif model_args.task_type == "translate-train-all":
        all_train_ds = []
        all_eval_ds = []
        for language in all_languages:
            train_ds = load_dataset("xnli", language, split="train")
            eval_ds = load_dataset("xnli", language, split="test")
            all_train_ds.append(train_ds.map(trans_func, batched=True, remove_columns=remove_columns))
            all_eval_ds.append(eval_ds.map(trans_func, batched=True, remove_columns=remove_columns))
        train_ds = XnliDataset(all_train_ds)
        eval_ds = XnliDataset(all_eval_ds)

    data_collator = DataCollatorWithPadding(tokenizer)

    num_labels = 3
    model = ErnieMForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=num_labels, classifier_dropout=model_args.classifier_dropout
    )
    n_layers = model.config.num_hidden_layers
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    warmup = training_args.warmup_steps if training_args.warmup_steps > 0 else training_args.warmup_ratio

    num_training_steps = (
        training_args.max_steps
        if training_args.max_steps > 0
        else len(train_ds) // training_args.train_batch_size * training_args.num_train_epochs
    )
    lr_scheduler = LinearDecayWithWarmup(training_args.learning_rate, num_training_steps, warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    # Construct dict
    name_dict = dict()
    for n, p in model.named_parameters():
        name_dict[p.name] = n

    simple_lr_setting = partial(layerwise_lr_decay, model_args.layerwise_decay, name_dict, n_layers)

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=training_args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=training_args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        lr_ratio=simple_lr_setting,
    )

    criterion = nn.CrossEntropyLoss()

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
        eval_dataset=eval_ds if training_args.do_eval else None,
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
        trainer.save_state()

    if training_args.do_eval:
        combined = {}
        for language in all_languages:
            test_ds = load_dataset("xnli", language, split="test")
            test_ds = test_ds.map(trans_func, batched=True, remove_columns=remove_columns, load_from_cache_file=True)
            metrics = trainer.evaluate(eval_dataset=test_ds)
            metrics = {k + f"_{language}": v for k, v in metrics.items()}
            combined.update({f"eval_accuracy_{language}": metrics.get(f"eval_accuracy_{language}", 0.0)})
            trainer.log_metrics("eval", metrics)

        combined.update({"eval_accuracy_average": np.mean(list(combined.values()))})
        trainer.log_metrics("eval", combined)
        trainer.save_metrics("eval", combined)


if __name__ == "__main__":
    do_train()
