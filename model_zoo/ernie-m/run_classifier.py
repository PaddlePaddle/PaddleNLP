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
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import paddle
from datasets import load_dataset
from paddle.io import Dataset
from paddle.metric import Accuracy

import paddlenlp
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ErnieMForSequenceClassification,
)
from paddlenlp.utils.log import logger

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
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    classifier_dropout: Optional[float] = field(default=0.1, metadata={"help": "Dropout rate."})
    layerwise_decay: Optional[float] = field(default=0.8, metadata={"help": "Layerwise decay ratio."})
    export_model_dir: Optional[str] = field(
        default="./best_models",
        metadata={"help": "Path to directory to store the exported inference model."},
    )
    use_test_data: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a tiny dataset for CI test."}
    )
    test_data_path: Optional[str] = field(default=None, metadata={"help": "Path to tiny dataset."})


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


def load_xnli_dataset(args, path, lang, split=None):
    """load dataset for specificed language"""
    if args.use_test_data:
        if args.test_data_path is None:
            raise ValueError("Should specified `test_data_path` for test datasets when `use_test_data` is True.")
        data_files = {
            "train": args.test_data_path,
            "validation": args.test_data_path,
            "test": args.test_data_path,
        }
        return load_dataset("json", data_files=data_files, split=split)
    else:
        return load_dataset(path, lang, split=split)


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

    set_seed(training_args.seed)

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

    # Dataset pre-process
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=model_args.max_seq_length)
    remove_columns = ["premise", "hypothesis"]

    def collect_all_languages_dataset(split):
        all_ds = []
        for language in all_languages:
            ds = load_xnli_dataset(model_args, "xnli", language, split=split)
            all_ds.append(ds.map(trans_func, batched=True, remove_columns=remove_columns))
        return XnliDataset(all_ds)

    if model_args.task_type == "cross-lingual-transfer":
        raw_datasets = load_xnli_dataset(model_args, "xnli", "en")
        if training_args.do_train:
            train_ds = raw_datasets["train"].map(trans_func, batched=True, remove_columns=remove_columns)
        if training_args.do_eval:
            eval_ds = raw_datasets["validation"].map(trans_func, batched=True, remove_columns=remove_columns)
        if training_args.do_predict:
            test_ds = raw_datasets["test"].map(trans_func, batched=True, remove_columns=remove_columns)
    elif model_args.task_type == "translate-train-all":
        if training_args.do_train:
            train_ds = collect_all_languages_dataset("train")
        if training_args.do_eval:
            eval_ds = collect_all_languages_dataset("validation")
        if training_args.do_predict:
            test_ds = collect_all_languages_dataset("test")
    else:
        raise ValueError(
            f"task_type should be 'cross-lingual-transfer' or 'translate-train-all' but '{model_args.task_type}' is specificed."
        )

    data_collator = DataCollatorWithPadding(tokenizer)

    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=num_labels, classifier_dropout=model_args.classifier_dropout
    )

    # Define the metrics of tasks.
    def compute_metrics(p):
        # Define the metrics of tasks.
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        metric = Accuracy()
        result = metric.compute(preds, label)
        metric.update(result)
        accu = metric.accumulate()
        return {"accuracy": accu}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=eval_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # optimizers=[optimizer, lr_scheduler],
    )

    def using_layerwise_lr_decay(layerwise_decay, model, training_args):
        """
        Generate parameter names needed to perform weight decay.
        All bias and LayerNorm parameters are excluded.
        """
        # params_list = [{"params": param, "learning_rate": lr * decay_ratio}, ... ]
        params_list = []
        n_layers = model.config.num_hidden_layers
        for name, param in model.named_parameters():
            ratio = 1.0
            param_to_train = {"params": param, "dygraph_key_name": name}
            if any(nd in name for nd in ["bias", "norm"]):
                param_to_train["weight_decay"] = 0.0
            else:
                param_to_train["weight_decay"] = training_args.weight_decay

            if "encoder.layers" in name:
                idx = name.find("encoder.layers.")
                layer = int(name[idx:].split(".")[2])
                ratio = layerwise_decay ** (n_layers - layer)
            elif "embedding" in name:
                ratio = layerwise_decay ** (n_layers + 1)

            param_to_train["learning_rate"] = ratio

            params_list.append(param_to_train)
        return params_list

    params_to_train = using_layerwise_lr_decay(model_args.layerwise_decay, model, training_args)

    trainer.set_optimizer_grouped_parameters(params_to_train)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluating
    if training_args.do_eval:
        combined = {}
        for language in all_languages:
            eval_ds = load_xnli_dataset(model_args, "xnli", language, split="validation")
            eval_ds = eval_ds.map(trans_func, batched=True, remove_columns=remove_columns, load_from_cache_file=True)
            metrics = trainer.evaluate(eval_dataset=eval_ds)
            metrics = {k + f"_{language}": v for k, v in metrics.items()}
            combined.update({f"eval_accuracy_{language}": metrics.get(f"eval_accuracy_{language}", 0.0)})
            trainer.log_metrics("eval", metrics)

        combined.update({"eval_accuracy_average": np.mean(list(combined.values()))})
        trainer.log_metrics("eval", combined)
        trainer.save_metrics("eval", combined)

    # Predicting
    if training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        trainer.log_metrics("test", test_ret.metrics)
        logits = test_ret.predictions
        max_value = np.max(logits, axis=1, keepdims=True)
        exp_data = np.exp(logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {"label": probs.argmax(axis=-1).tolist(), "confidence": probs.max(axis=-1).tolist()}
        out_file = open(os.path.join(training_args.output_dir, "test_results.json"), "w")
        json.dump(out_dict, out_file)

    # Export inference model
    if training_args.do_export and paddle.distributed.get_rank() == 0:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        model_to_save = trainer.model
        model_to_save = model_to_save._layers if isinstance(model_to_save, paddle.DataParallel) else model_to_save
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        ]
        model_args.export_model_dir = os.path.join(model_args.export_model_dir, "export")
        paddlenlp.transformers.export_model(
            model=model_to_save, input_spec=input_spec, path=model_args.export_model_dir
        )
        trainer.tokenizer.save_pretrained(model_args.export_model_dir)


if __name__ == "__main__":
    do_train()
