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

from dataclasses import dataclass, field
from typing import List, Optional

import paddle
import yaml

from paddlenlp.trainer import PredictionOutput, Trainer


def load_config(config_file_path, task_name, dataset_name, model_args, data_args, training_args):
    config = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)
    # Set the batch size of trainer setting

    config = config[task_name][dataset_name]
    for args in (model_args, data_args, training_args):
        for arg in config.keys():
            if hasattr(args, arg):
                setattr(args, arg, config[arg])
    return model_args, data_args, training_args


def get_dynamic_max_length(examples, default_max_length: int, dynamic_max_length: List[int]) -> int:
    """get max_length by examples which you can change it by examples in batch"""
    # if the input is a batch of examples
    if isinstance(examples["input_ids"][0], list):
        cur_length = max([len(i) for i in examples["input_ids"]])
    # if the input is a single example
    else:
        cur_length = len(examples["input_ids"])

    max_length = default_max_length
    for max_length_option in sorted(dynamic_max_length):
        if cur_length <= max_length_option:
            max_length = max_length_option
            break
    return max_length


def prepare_train_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function.
    contexts = examples["context"]
    questions = examples["question"]

    if args.dynamic_max_length is not None:
        tokenized_examples = tokenizer(
            questions, contexts, stride=args.doc_stride, max_length=args.max_seq_length, truncation=True
        )
        max_length = get_dynamic_max_length(
            examples=tokenized_examples,
            default_max_length=args.max_seq_length,
            dynamic_max_length=args.dynamic_max_length,
        )
        # always pad to max_length
        tokenized_examples = tokenizer(
            questions, contexts, stride=args.doc_stride, max_length=max_length, padding="max_length", truncation=True
        )
    else:
        tokenized_examples = tokenizer(
            questions, contexts, stride=args.doc_stride, max_length=args.max_seq_length, truncation=True
        )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples["token_type_ids"][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HuggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = examples["context"]
    questions = examples["question"]

    if args.dynamic_max_length is not None:
        tokenized_examples = tokenizer(
            questions, contexts, stride=args.doc_stride, max_length=args.max_seq_length, truncation=True
        )
        max_length = get_dynamic_max_length(
            examples=tokenized_examples,
            default_max_length=args.max_seq_length,
            dynamic_max_length=args.dynamic_max_length,
        )
        # always pad to max_length
        tokenized_examples = tokenizer(
            questions, contexts, stride=args.doc_stride, max_length=max_length, padding="max_length", truncation=True
        )
    else:
        tokenized_examples = tokenizer(
            questions, contexts, stride=args.doc_stride, max_length=args.max_seq_length, truncation=True
        )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples["token_type_ids"][i]
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index and k != len(sequence_ids) - 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(input=start_logits, label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)


# Data pre-process function for clue benchmark datatset
def seq_convert_example(
    example, label_list, tokenizer=None, max_seq_length=512, dynamic_max_length: Optional[List[int]] = None, **kwargs
):
    """convert a glue example into necessary features"""
    is_test = False
    if "label" not in example.keys():
        is_test = True

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        example["label"] = int(example["label"]) if label_dtype != "float32" else float(example["label"])
        label = example["label"]
    # Convert raw text to feature
    if "keyword" in example:  # CSL
        sentence1 = " ".join(example["keyword"])
        example = {"sentence1": sentence1, "sentence2": example["abst"], "label": example["label"]}
    elif "target" in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = (
            example["text"],
            example["target"]["span1_text"],
            example["target"]["span2_text"],
            example["target"]["span1_index"],
            example["target"]["span2_index"],
        )
        text_list = list(text)
        assert text[pronoun_idx : (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx : (query_idx + len(query))] == query, "query: {}".format(query)
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
        example["sentence"] = text

    if tokenizer is None:
        return example
    if "sentence" in example:
        if dynamic_max_length is not None:
            temp_example = tokenizer(example["sentence"], max_length=max_seq_length, truncation=True)
            max_length = get_dynamic_max_length(
                examples=temp_example, default_max_length=max_seq_length, dynamic_max_length=dynamic_max_length
            )
            # always pad to max_length
            example = tokenizer(example["sentence"], max_length=max_length, padding="max_length", truncation=True)
        else:
            example = tokenizer(example["sentence"], max_length=max_seq_length, truncation=True)
    elif "sentence1" in example:
        if dynamic_max_length is not None:
            temp_example = tokenizer(
                example["sentence1"],
                text_pair=example["sentence2"],
                max_length=max_seq_length,
                truncation=True,
            )
            max_length = get_dynamic_max_length(
                examples=temp_example, default_max_length=max_seq_length, dynamic_max_length=dynamic_max_length
            )
            example = tokenizer(
                example["sentence1"],
                text_pair=example["sentence2"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
        else:
            example = tokenizer(
                example["sentence1"],
                text_pair=example["sentence2"],
                max_length=max_seq_length,
                truncation=True,
            )

    if not is_test:
        if "token_type_ids" in example:
            return {"input_ids": example["input_ids"], "token_type_ids": example["token_type_ids"], "labels": label}
        else:
            return {"input_ids": example["input_ids"], "labels": label}
    else:
        return {"input_ids": example["input_ids"], "token_type_ids": example["token_type_ids"]}


def token_convert_example(
    example,
    tokenizer,
    no_entity_id,
    max_seq_length=512,
    return_length=False,
    dynamic_max_length: Optional[List[int]] = None,
):
    if "labels" in example:
        labels = example["labels"]
        example = example["tokens"]
        if dynamic_max_length is not None:
            tokenized_input = tokenizer(
                example,
                is_split_into_words=True,
                max_length=max_seq_length,
                truncation=True,
                return_length=return_length,
            )
            max_length = get_dynamic_max_length(
                examples=tokenized_input, default_max_length=max_seq_length, dynamic_max_length=dynamic_max_length
            )
            # always pad to max_length
            tokenized_input = tokenizer(
                example,
                is_split_into_words=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_length=return_length,
            )
        else:
            tokenized_input = tokenizer(
                example,
                is_split_into_words=True,
                max_length=max_seq_length,
                truncation=True,
                return_length=return_length,
            )

        # -2 for [CLS] and [SEP]
        if len(tokenized_input["input_ids"]) - 2 < len(labels):
            labels = labels[: len(tokenized_input["input_ids"]) - 2]
        tokenized_input["labels"] = [no_entity_id] + labels + [no_entity_id]
        tokenized_input["labels"] += [no_entity_id] * (
            len(tokenized_input["input_ids"]) - len(tokenized_input["labels"])
        )
    else:
        if example["tokens"] == []:
            if return_length:
                tokenized_input = {"labels": [], "input_ids": [], "token_type_ids": [], "length": 0, "seq_len": 0}
            else:
                tokenized_input = {"labels": [], "input_ids": [], "token_type_ids": []}

            return tokenized_input
        if dynamic_max_length is not None:
            tokenized_input = tokenizer(
                example["tokens"],
                max_length=max_seq_length,
                truncation=True,
                is_split_into_words=True,
                return_length=return_length,
            )
            max_length = get_dynamic_max_length(
                examples=tokenized_input, default_max_length=max_seq_length, dynamic_max_length=dynamic_max_length
            )
            # always pad to max_length
            tokenized_input = tokenizer(
                example["tokens"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                is_split_into_words=True,
                return_length=return_length,
            )
        else:
            tokenized_input = tokenizer(
                example["tokens"],
                max_length=max_seq_length,
                truncation=True,
                is_split_into_words=True,
                return_length=return_length,
            )

        label_ids = example["ner_tags"]
        if len(tokenized_input["input_ids"]) - 2 < len(label_ids):
            label_ids = label_ids[: len(tokenized_input["input_ids"]) - 2]
        label_ids = [no_entity_id] + label_ids + [no_entity_id]

        label_ids += [no_entity_id] * (len(tokenized_input["input_ids"]) - len(label_ids))
        tokenized_input["labels"] = label_ids
    return tokenized_input


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset: str = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    # Additional configs for QA task.
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )

    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate in the nbest_predictions.json output file."
        },
    )

    max_query_length: int = field(
        default=64,
        metadata={"help": "Max query length."},
    )

    max_answer_length: int = field(
        default=30,
        metadata={"help": "Max answer length."},
    )

    dynamic_max_length: Optional[List[int]] = field(
        default=None,
        metadata={"help": "dynamic max length from batch, it can be array of length, eg: 16 32 64 128"},
    )

    do_lower_case: bool = field(
        default=False,
        metadata={
            "help": "Whether to lower case the input text. Should be True for uncased models and False for cased models."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )

    # TODO(wj-Mcat): support padding configuration: `max_length`, `longest_first`


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
    config: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    export_model_dir: Optional[str] = field(
        default="./best_models",
        metadata={"help": "Path to directory to store the exported inference model."},
    )
