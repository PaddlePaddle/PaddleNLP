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

import collections
import os
from functools import partial

import datasets
import paddle
from data_collator import DataCollator
from datasets import load_dataset
from finetune_args import DataArguments, ModelArguments
from layout_trainer import LayoutTrainer
from utils import PostProcessor, PreProcessor, anls_score

from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import AutoModelForQuestionAnswering, AutoTokenizer
from paddlenlp.utils.log import logger


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

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

    train_ds, dev_ds, test_ds = load_dataset(data_args.dataset_name, split=["train", "validation", "test"])

    if training_args.do_train:
        column_names = train_ds.column_names
    elif training_args.do_eval:
        column_names = dev_ds.column_names
    elif training_args.do_predict:
        column_names = test_ds.column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        raise NotImplementedError

    num_labels = 2

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, num_classes=num_labels)
    model.config["has_visual_segment_embedding"] = False

    preprocessor = PreProcessor()
    postprocessor = PostProcessor()
    training_args.label_names = ["start_positions", "end_positions"]

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    preprocess_func = partial(
        preprocessor.preprocess_mrc,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=data_args.doc_stride,
        max_size=data_args.target_size,
        target_size=data_args.target_size,
        use_segment_box=data_args.use_segment_box,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        is_training=True,
        lang=data_args.lang,
    )

    preprocess_func_for_valid = partial(
        preprocessor.preprocess_mrc,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=data_args.doc_stride,
        max_size=data_args.target_size,
        target_size=data_args.target_size,
        use_segment_box=data_args.use_segment_box,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        is_training=False,
        lang=data_args.lang,
    )

    postprocess_func = partial(postprocessor.postprocess_mrc, tokenizer=tokenizer, lang=data_args.lang)

    # Dataset pre-process
    if training_args.do_train:
        if data_args.train_nshard > 1:
            logger.info(f"spliting train dataset into {data_args.train_nshard} shard")
            train_shards = []
            for idx in range(data_args.train_nshard):
                train_shards.append(
                    train_ds.shard(num_shards=data_args.train_nshard, index=idx).map(
                        preprocess_func,
                        batched=True,
                        remove_columns=column_names,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                    )
                )
            train_dataset = datasets.concatenate_datasets(train_shards)
        else:
            train_dataset = train_ds.map(
                preprocess_func,
                batched=True,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_eval:
        eval_dataset = dev_ds.map(
            preprocess_func_for_valid,
            batched=True,
            remove_columns=column_names,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_predict:
        test_dataset = test_ds.map(
            preprocess_func_for_valid,
            batched=True,
            remove_columns=column_names,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    data_collator = DataCollator(
        tokenizer, padding="max_length", label_pad_token_id=-100, max_length=max_seq_length, return_tensors="pd"
    )

    def compute_metrics(eval_preds):
        def _convert(examples):
            """Convert to evaluation data format"""
            formatted_examples = []
            for example in examples:
                formatted_example = {}
                formatted_example["id"] = example["id"]
                formatted_example["annotations"] = {
                    "qid": [],
                    "question": [],
                    "value": [],
                }
                for i in range(len(example["annotations"])):
                    formatted_example["annotations"]["qid"].append(example["annotations"][i]["qid"])
                    formatted_example["annotations"]["question"].append(example["annotations"][i]["question"])
                    formatted_example["annotations"]["value"].append(example["annotations"][i]["value"])
                formatted_examples.append(formatted_example)
            return formatted_examples

        pred_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        ref_dict = collections.defaultdict(lambda: collections.defaultdict(list))

        preds = _convert(eval_preds.predictions)
        labels = _convert(eval_preds.label_ids)

        for pred in preds:
            for key, values in zip(pred["annotations"]["qid"], pred["annotations"]["value"]):
                pred_dict[pred["id"]][key].extend(values)
        for label in labels:
            for key, values in zip(label["annotations"]["qid"], label["annotations"]["value"]):
                ref_dict[label["id"]][key].extend(values)
        score = anls_score(ref_dict, pred_dict)
        return score

    trainer = LayoutTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        post_process_function=postprocess_func,
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

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        postprocessor.examples_cache = collections.defaultdict(list)
        postprocessor.features_cache = collections.defaultdict(list)
        metrics = trainer.predict(test_dataset, test_ds)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
