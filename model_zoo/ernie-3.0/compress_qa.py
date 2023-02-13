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

from functools import partial

import paddle
import paddle.nn.functional as F
from datasets import load_dataset
from utils import (
    DataArguments,
    ModelArguments,
    QuestionAnsweringTrainer,
    load_config,
    prepare_train_features,
    prepare_validation_features,
)

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.metrics.squad import compute_prediction
from paddlenlp.trainer import CompressionArguments, EvalPrediction, PdArgumentParser
from paddlenlp.transformers import ErnieForQuestionAnswering, ErnieTokenizer


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses()

    # Load model and data config
    model_args, data_args, compression_args = load_config(
        model_args.config, "QuestionAnswering", data_args.dataset, model_args, data_args, compression_args
    )

    paddle.set_device(compression_args.device)
    data_args.dataset = data_args.dataset.strip()

    # Log model and data config
    compression_args.print_config(model_args, "Model")
    compression_args.print_config(data_args, "Data")

    raw_datasets = load_dataset("clue", data_args.dataset)

    label_list = getattr(raw_datasets["train"], "label_list", None)
    data_args.label_list = label_list

    # Define tokenizer, model, loss function.
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)
    model = ErnieForQuestionAnswering.from_pretrained(model_args.model_name_or_path)

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.

    raw_datasets = load_dataset("clue", data_args.dataset)
    column_names = raw_datasets["train"].column_names
    label_list = getattr(raw_datasets["train"], "label_list", None)
    data_args.label_list = label_list
    # Create train feature from dataset
    with compression_args.main_process_first(desc="train dataset map pre-processing"):
        # Dataset pre-process
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            partial(prepare_train_features, tokenizer=tokenizer, args=data_args),
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    with compression_args.main_process_first(desc="evaluate dataset map pre-processing"):
        eval_examples = raw_datasets["validation"]
        eval_dataset = eval_examples.map(
            partial(prepare_validation_features, tokenizer=tokenizer, args=data_args),
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # Define data collector
    data_collator = DataCollatorWithPadding(tokenizer)

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, all_nbest_json, scores_diff_json = compute_prediction(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
        )

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=predictions, label_ids=references)

    def criterion(outputs, label):
        start_logits, end_logits = outputs
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = F.cross_entropy(input=start_logits, label=start_position)
        end_loss = F.cross_entropy(input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=compression_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        tokenizer=tokenizer,
        criterion=criterion,
    )

    compression_args.print_config()

    trainer.compress()


if __name__ == "__main__":
    main()
