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
import sys
from functools import partial
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    Trainer,
)
from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from paddlenlp.utils.log import logger


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to 
    specify them on the command line.
    """

    dataset: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        })

    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to lower case the input text. Should be True for uncased models and False for cased models."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the dataset cache."},
    )
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory to store the exported inference model."
        },
    )


# Data pre-process function for clue benchmark datatset
def convert_clue(example,
                 label_list,
                 tokenizer=None,
                 max_seq_length=512,
                 **kwargs):
    """convert a glue example into necessary features"""
    is_test = False
    if 'label' not in example.keys():
        is_test = True

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # print("label_list", label_list)
        # Get the label
        # example['label'] = np.array(example["label"], dtype="int64")
        example['label'] = int(
            example["label"]) if label_dtype != "float32" else float(
                example["label"])
        label = example['label']
    # Convert raw text to feature
    if 'keyword' in example:  # CSL
        sentence1 = " ".join(example['keyword'])
        example = {
            'sentence1': sentence1,
            'sentence2': example['abst'],
            'label': example['label']
        }
    elif 'target' in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
            'target']['span1_text'], example['target']['span2_text'], example[
                'target']['span1_index'], example['target']['span2_index']
        text_list = list(text)
        assert text[pronoun_idx:(
            pronoun_idx +
            len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx +
                               len(query))] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example['sentence'] = text

    if tokenizer is None:
        return example
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(example['sentence1'],
                            text_pair=example['sentence2'],
                            max_seq_len=max_seq_length)

    if not is_test:
        return {
            "input_ids": example['input_ids'],
            "token_type_ids": example['token_type_ids'],
            "labels": label
        }
    else:
        return {
            "input_ids": example['input_ids'],
            "token_type_ids": example['token_type_ids']
        }


def clue_trans_fn(example, tokenizer, args):
    return convert_clue(example,
                        tokenizer=tokenizer,
                        label_list=args.label_list,
                        max_seq_length=args.max_seq_length)


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(
                training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    data_args.dataset = data_args.dataset.strip()

    dataset_config = data_args.dataset.split(" ")
    print(dataset_config)
    raw_datasets = load_dataset(
        dataset_config[0],
        name=None if len(dataset_config) <= 1 else dataset_config[1],
        splits=('train', 'dev'))

    data_args.label_list = getattr(raw_datasets['train'], "label_list", None)
    num_classes = 1 if raw_datasets["train"].label_list == None else len(
        raw_datasets['train'].label_list)

    # Define tokenizer, model, loss function.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_classes=num_classes)
    criterion = nn.loss.CrossEntropyLoss(
    ) if data_args.label_list else nn.loss.MSELoss()

    # Define dataset pre-process function
    trans_fn = partial(clue_trans_fn, tokenizer=tokenizer, args=data_args)

    # Define data collector
    data_collator = DataCollatorWithPadding(tokenizer)

    # Dataset pre-process
    if training_args.do_train:
        train_dataset = raw_datasets["train"].map(trans_fn)
    if training_args.do_eval:
        eval_dataset = raw_datasets["dev"].map(trans_fn)
    if training_args.do_predict:
        test_dataset = raw_datasets["test"].map(trans_fn)

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        probs = F.softmax(preds, axis=1)
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
        if test_ret.label_ids is None:
            paddle.save(
                test_ret.predictions,
                os.path.join(training_args.output_dir, "test_results.pdtensor"),
            )

    # export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64")  # segment_ids
        ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir,
                                                       "export")
        paddlenlp.transformers.export_model(model=trainer.model,
                                            input_spec=input_spec,
                                            path=model_args.export_model_dir)


if __name__ == "__main__":
    main()
