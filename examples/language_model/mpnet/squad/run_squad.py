# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
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
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import numpy as np
import paddle
from datasets import load_dataset
from paddle.io import DataLoader, Dataset

from paddlenlp.data import Pad, Stack
from paddlenlp.metrics.squad import compute_prediction, squad_evaluate
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments, set_seed
from paddlenlp.trainer.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    IterableDatasetShard,
    find_batch_size,
    has_length,
)
from paddlenlp.trainer.utils.helper import (
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from paddlenlp.transformers import (
    BertForQuestionAnswering,
    BertTokenizer,
    ErnieForQuestionAnswering,
    ErnieTokenizer,
    MPNetForQuestionAnswering,
    MPNetTokenizer,
)
from paddlenlp.utils.batch_sampler import (
    DistributedBatchSampler as NlpDistributedBatchSampler,
)
from paddlenlp.utils.log import logger


def is_datasets_available():
    import importlib

    return importlib.util.find_spec("datasets") is not None


class MPNetTrainer(Trainer):
    def set_eval_collator(self, collator):
        self.eval_collate_fn = collator

    def set_eval_raw_dataset(self, raw_dataset):
        self.eval_raw_dataset = raw_dataset

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~paddle.io.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`paddle.io.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self._is_iterable_dataset(eval_dataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.eval_collate_fn,
                num_workers=self.args.dataloader_num_workers,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=self.eval_collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self.model

        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase):
            # support for inner dataloader
            batch_size = dataloader._batch_sampler.batch_size
            # alias for inner dataloader
            dataloader.dataset = dataloader._dataset
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        num_samples = None
        if max_eval_iters > 0:
            # on eval limit steps
            num_samples = batch_size * self.args.world_size * max_eval_iters
            if isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase) and isinstance(
                dataloader._batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    ((self.state.global_step) // args.eval_steps)
                    * max_eval_iters
                    * args.per_device_eval_batch_size
                    * args.world_size
                )
                dataloader._batch_sampler.set_epoch(consumed_samples=consumed_samples)

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")
            else:
                logger.info(f"  Total prediction steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.world_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        losses = []
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                # losses = self._nested_gather(loss.repeat(batch_size))
                # losses = self._nested_gather(loss)
                losses = self._nested_gather(paddle.tile(loss, repeat_times=[batch_size, 1]))
                losses_host = losses if losses_host is None else paddle.concat((losses_host, losses), axis=0)

            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if num_samples is not None:
            pass
        elif has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        model.train()

        if self.compute_metrics is not None and all_preds is not None:
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels),
                data_loader=dataloader,
                raw_dataset=self.eval_raw_dataset,
            )
        else:
            metrics = {}

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
    "mpnet": (MPNetForQuestionAnswering, MPNetTokenizer),
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
    doc_stride: Optional[int] = field(
        default=128,
        metadata={"help": ("When splitting up a long document into chunks, how much stride to take between chunks.")},
    )
    n_best_size: Optional[int] = field(
        default=20,
        metadata={
            "help": ("The total number of n-best predictions to generate in the nbest_predictions.json output file.")
        },
    )
    null_score_diff_threshold: Optional[float] = field(
        default=0.0,
        metadata={"help": ("If null_score - best_non_null is greater than the threshold predict null.")},
    )
    max_query_length: Optional[int] = field(
        default=64,
        metadata={"help": ("Max query length.")},
    )
    max_answer_length: Optional[int] = field(
        default=30,
        metadata={"help": ("Max answer length.")},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to lower case the input text. Should be True for uncased models and False for cased models."
            )
        },
    )
    verbose: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to output verbose log.")},
    )
    version_2_with_negative: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If true, the SQuAD examples contain some that do not have an answer.",
                "If using squad v2.0, it should be set true.",
            )
        },
    )


@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Train data path."},
    )
    predict_file: Optional[str] = field(
        default=None,
        metadata={"help": "Predict data path."},
    )


def _get_layer_lr_radios(layer_decay=0.8, n_layers=12):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = OrderedDict(
        {
            "mpnet.embeddings.": 0,
            "mpnet.encoder.relative_attention_bias.": 0,
            "qa_outputs.": n_layers + 2,
        }
    )
    for layer in range(n_layers):
        key_to_depths[f"mpnet.encoder.layer.{str(layer)}."] = layer + 1
    return {key: (layer_decay ** (n_layers + 2 - depth)) for key, depth in key_to_depths.items()}


def prepare_train_features(examples, tokenizer, args):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    contexts = examples["context"]
    questions = examples["question"]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        questions, contexts, max_length=args.max_seq_length, stride=args.doc_stride, return_attention_mask=True
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
        sequence_A_lengths = input_ids.index(tokenizer.sep_token_id) + 2
        sequence_B_lengths = len(input_ids) - sequence_A_lengths
        sequence_ids = [0] * sequence_A_lengths + [1] * sequence_B_lengths

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
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = examples["context"]
    questions = examples["question"]

    tokenized_examples = tokenizer(
        questions, contexts, stride=args.doc_stride, max_length=args.max_seq_length, return_attention_mask=True
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        input_ids = tokenized_examples["input_ids"][i]
        sequence_A_lengths = input_ids.index(tokenizer.sep_token_id) + 2
        sequence_B_lengths = len(input_ids) - sequence_A_lengths
        sequence_ids = [0] * sequence_A_lengths + [1] * sequence_B_lengths
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


@dataclass
class TrainDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = []
        start_positions = []
        end_positions = []
        batch = {}

        for feature in features:
            input_ids.append(feature["input_ids"])
            start_positions.append(feature["start_positions"])
            end_positions.append(feature["end_positions"])

        input_ids = (Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(input_ids),)  # input_ids
        start_positions = (Stack(dtype="int64")(start_positions),)  # start_positions
        end_positions = (Stack(dtype="int64")(end_positions),)  # end_positions

        batch["input_ids"] = input_ids[0]
        batch["start_positions"] = start_positions[0]
        batch["end_positions"] = end_positions[0]

        return batch


@dataclass
class EvalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = []
        batch = {}

        for feature in features:
            input_ids.append(feature["input_ids"])

        input_ids = (Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(input_ids),)  # input_ids

        batch["input_ids"] = input_ids[0]

        return batch


def compute_metrics(eval_preds, data_loader=None, raw_dataset=None, model_args=None):
    start_logits, end_logits = eval_preds.predictions
    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        raw_dataset,
        data_loader.dataset,
        (start_logits, end_logits),
        model_args.version_2_with_negative,
        model_args.n_best_size,
        model_args.max_answer_length,
        model_args.null_score_diff_threshold,
    )
    squad_evaluate(examples=[raw_data for raw_data in raw_dataset], preds=all_predictions, na_probs=scores_diff_json)
    return {}


@paddle.no_grad()
def evaluate(model, data_loader, raw_dataset, args, global_step, write_predictions=False):
    model.eval()

    all_start_logits = []
    all_end_logits = []

    for batch in data_loader:
        input_ids = batch[0]
        start_logits_tensor, end_logits_tensor = model(input_ids)

        for idx in range(start_logits_tensor.shape[0]):
            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        raw_dataset,
        data_loader.dataset,
        (all_start_logits, all_end_logits),
        args.version_2_with_negative,
        args.n_best_size,
        args.max_answer_length,
        args.null_score_diff_threshold,
    )

    # Can also write all_nbest_json and scores_diff_json files if needed
    if write_predictions:
        with open(f"{str(global_step)}_prediction.json", "w", encoding="utf-8") as writer:
            writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

    squad_evaluate(examples=[raw_data for raw_data in raw_dataset], preds=all_predictions, na_probs=scores_diff_json)

    model.train()


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


def do_train():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    model_args.model_type = model_args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)
    model = model_class.from_pretrained(model_args.model_name_or_path)

    if training_args.do_train:
        # layer_lr for base
        if model_args.layer_lr_decay != 1.0:
            layer_lr_radios_map = _get_layer_lr_radios(model_args.layer_lr_decay, n_layers=12)
            for name, parameter in model.named_parameters():
                layer_lr_radio = 1.0
                for k, radio in layer_lr_radios_map.items():
                    if k in name:
                        layer_lr_radio = radio
                        break
                parameter.optimize_attr["learning_rate"] *= layer_lr_radio

        if model_args.version_2_with_negative:
            train_examples = load_dataset("squad_v2", split="train")
        else:
            train_examples = load_dataset("squad", split="train")
        column_names = train_examples.column_names
        train_ds = train_examples.map(
            partial(prepare_train_features, tokenizer=tokenizer, args=model_args),
            batched=True,
            remove_columns=column_names,
            num_proc=4,
        )

    if training_args.do_eval:
        if model_args.version_2_with_negative:
            dev_examples = load_dataset("squad_v2", split="validation")
        else:
            dev_examples = load_dataset("squad", split="validation")
        column_names = dev_examples.column_names
        dev_ds = dev_examples.map(
            partial(prepare_validation_features, tokenizer=tokenizer, args=model_args),
            batched=True,
            remove_columns=column_names,
            num_proc=4,
        )

    batchify_fn_train = TrainDataCollator(tokenizer)
    batchify_fn_eval = EvalDataCollator(tokenizer)
    criterion = CrossEntropyLossForSQuAD()

    compute_metrics_func = partial(
        compute_metrics,
        model_args=model_args,
    )

    trainer = MPNetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=batchify_fn_train,
        criterion=criterion,
        compute_metrics=compute_metrics_func,
    )
    trainer.set_eval_collator(batchify_fn_eval)
    trainer.set_eval_raw_dataset(dev_examples)

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
