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
"""
ERNIE-1.0 pretraining scripts.
"""
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paddle
from data_tools.dataset_utils import build_train_valid_test_datasets

from paddlenlp.data import Stack
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
    speed_metrics,
)
from paddlenlp.transformers import (
    ErnieConfig,
    ErnieForMaskedLM,
    ErnieForPretraining,
    ErniePretrainingCriterion,
    ErnieTokenizer,
    LinearAnnealingWithWarmupDecay,
)
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "ernie": (
        ErnieConfig,
        ErnieForPretraining,
        ErniePretrainingCriterion,
        ErnieTokenizer,
    ),
}

from paddlenlp.trainer.utils.doc import add_start_docstrings


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class PreTrainingArguments(TrainingArguments):
    min_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate deacyed to."},
    )
    decay_steps: float = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, will use the min_learning_rate."
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluating.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    input_dir: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    split: str = field(default="949,50,1", metadata={"help": "Train/valid/test data split."})

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    masked_lm_prob: float = field(
        default=0.15,
        metadata={"help": "Mask token prob."},
    )
    short_seq_prob: float = field(
        default=0.1,
        metadata={"help": "Short sequence prob."},
    )
    share_folder: bool = field(
        default=False,
        metadata={"help": "Use share folder for data dir and output dir on multi machine."},
    )
    favor_longer_ngram: bool = field(
        default=False,
        metadata={"help": "Whether to favor long ngrams"},
    )
    max_ngrams: int = field(
        default=3,
        metadata={"help": "Max N Grams"},
    )
    data_impl: str = field(
        default="mmap",
        metadata={"help": "mmap/lazy format converted from json"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to pre-train from.
    """

    model_type: Optional[str] = field(
        default="ernie", metadata={"help": "Only support for ernie pre-training for now."}
    )
    model_name_or_path: str = field(
        default="ernie-1.0",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    binary_head: Optional[bool] = field(default=True, metadata={"help": "True for NSP task."})
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "The hidden dropout prob."})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"help": "The attention probs dropout prob."})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


def create_pretrained_dataset(
    data_args,
    training_args,
    data_file,
    tokenizer,
    binary_head=True,
):

    train_valid_test_num_samples = [
        training_args.per_device_train_batch_size
        * training_args.world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps,
        training_args.per_device_eval_batch_size
        * training_args.world_size
        * training_args.eval_iters
        * (training_args.max_steps // training_args.eval_steps + 1),
        training_args.per_device_eval_batch_size * training_args.world_size * training_args.test_iters,
    ]
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=data_file,
        args=data_args,
        tokenizer=tokenizer,
        splits_string=data_args.split,
        train_valid_test_num_samples=train_valid_test_num_samples,
        max_seq_length=data_args.max_seq_length,
        masked_lm_prob=data_args.masked_lm_prob,
        short_seq_prob=data_args.short_seq_prob,
        seed=training_args.seed,
        skip_warmup=True,
        binary_head=binary_head,
        max_seq_length_dec=None,
        dataset_type="ernie",
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode")
        input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_labels, next_sentence_labels = data
        if tokenizer.pad_token_id in input_ids:
            input_ids = input_ids[0 : list(input_ids).index(tokenizer.pad_token_id)]
        logger.info(tokenizer._decode(input_ids))
        for pos, label in zip(masked_lm_positions, masked_lm_labels):
            input_ids[pos] = label
        logger.info(tokenizer._decode(input_ids))
        logger.info(tokenizer.convert_ids_to_tokens(masked_lm_labels))

    print_dataset(train_ds[0], "train")
    print_dataset(valid_ds[0], "valid")
    print_dataset(test_ds[0], "test")

    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        # 0. input_ids,
        # 1. segment_ids,
        # 2. input_mask,
        # 3. masked_lm_positions,
        # 4. masked_lm_labels,
        # 5. next_sentence_labels
        for i in (0, 1, 2, 5):
            out[i] = stack_fn([x[i] for x in data])
        out[5] = out[5].reshape([-1, 1])
        _, seq_length = out[0].shape
        size = sum(len(x[3]) for x in data)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        if size % 8 != 0:
            size += 8 - (size % 8)
        out[3] = np.full(size, 0, dtype=np.int32)
        # masked_lm_labels
        out[4] = np.full([size, 1], -1, dtype=np.int64)
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[3]):
                out[3][mask_token_num] = i * seq_length + pos
                out[4][mask_token_num] = x[4][j]
                mask_token_num += 1

        return {
            "input_ids": out[0],
            "token_type_ids": out[1],
            "attention_mask": out[2],
            "masked_positions": out[3],
            "labels": (out[4], out[5]),
        }

    return train_ds, valid_ds, test_ds, _collate_data


def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return args.input_dir.split()
    else:
        files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and "_idx.npz" in str(f))
        ]
        files = [x.replace("_idx.npz", "") for x in files]

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret

    return files


def set_seed(args):
    if args.device == "cpu":
        idx = 0
    else:
        idx = paddle.distributed.get_rank()
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    paddle.seed(args.seed + idx)


class PretrainingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataloader = getattr(self, "eval_dataloader", None)
        if eval_dataloader is None:
            eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            # must call data loader, otherwise, it will init many times, cause OOM error.
            self.eval_dataloader = eval_dataloader()

        start_time = time.time()
        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        eval_loop = self.evaluation_loop

        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            # Only evaluate max_eval_iters
            max_eval_iters=self.args.eval_iters,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
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
        return output.metrics

    def _get_eval_sampler(self, eval_dataset) -> Optional[paddle.io.Sampler]:
        return DistributedBatchSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
        )

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
        )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    set_seed(training_args)
    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    training_args.eval_iters = 10
    training_args.test_iters = training_args.eval_iters * 10

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # if last_checkpoint is None and len(
        #         os.listdir(training_args.output_dir)) > 1:
        #     raise ValueError(
        #         f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        #         "Use --overwrite_output_dir to overcome.")
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    config_class, model_class, criterion_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

    if model_args.binary_head is False:
        model_class = ErnieForMaskedLM

    pretrained_models_list = list(model_class.pretrained_init_configuration.keys())

    if model_args.model_name_or_path in pretrained_models_list:
        logger.warning(f"Your model {model_args.model_name_or_path} is training from scratch !!!")
        model_config = model_class.pretrained_init_configuration[model_args.model_name_or_path]
        model_config["hidden_dropout_prob"] = model_args.hidden_dropout_prob
        model_config["attention_probs_dropout_prob"] = model_args.attention_probs_dropout_prob
        model = model_class(config_class(**model_config))
        # model_config["enable_recompute"] = args.use_recompute
    else:
        logger.warning(f"Your model is continue training from {model_args.model_name_or_path}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        )

    class CriterionWrapper(paddle.nn.Layer):
        """ """

        def __init__(self):
            """CriterionWrapper"""
            super(CriterionWrapper, self).__init__()
            self.criterion = criterion_class(with_nsp_loss=model_args.binary_head)

        def forward(self, output, labels):
            """forward function

            Args:
                output (tuple): prediction_scores, seq_relationship_score
                labels (tuple): masked_lm_labels, next_sentence_labels

            Returns:
                Tensor: final loss.
            """
            masked_lm_labels, next_sentence_labels = labels
            if model_args.binary_head:
                prediction_scores, seq_relationship_score = output

                lm_loss, sop_loss = self.criterion(
                    prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels
                )

                loss = lm_loss + sop_loss

            else:
                prediction_scores = output
                loss = self.criterion(prediction_scores, None, masked_lm_labels)

            return loss

    # Create the learning_rate sheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps
    warmup_steps = training_args.warmup_ratio * training_args.max_steps

    lr_scheduler = LinearAnnealingWithWarmupDecay(
        training_args.learning_rate,
        training_args.min_learning_rate,
        warmup_step=warmup_steps,
        decay_step=training_args.decay_steps,
    )

    data_file = get_train_data_file(data_args)
    tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name_or_path)
    tokenizer.extend_chinese_char()

    train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(
        data_args, training_args, data_file, tokenizer, model_args.binary_head
    )

    trainer = PretrainingTrainer(
        model=model,
        criterion=CriterionWrapper(),
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(None, lr_scheduler),
        tokenizer=tokenizer,
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

    if training_args.do_predict:
        test_ret = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_ret.metrics)


if __name__ == "__main__":
    main()
