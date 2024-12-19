# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
GPT/Ernie35 pretraining scripts.
"""
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paddle
from ernie_dataset import build_train_valid_test_datasets, print_rank_0
from modeling import Ernie35Config, Ernie35ForCausalLM
from tokenizer import Ernie35Tokenizer

from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
    speed_metrics,
)
from paddlenlp.transformers import (  # Ernie35Config,; Ernie35ForCausalLM,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
)
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "ernie": (
        Ernie35Config,
        Ernie35ForCausalLM,
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
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    share_folder: bool = field(
        default=False,
        metadata={"help": "Use share folder for data dir and output dir on multi machine."},
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
        default="__internal_testing__/tiny-random-ernie",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "use_flash_attention"},
    )
    use_fused_ln: bool = field(
        default=False,
        metadata={"help": "ernie, use_fused_ln"},
    )
    fuse_attention_qkv: bool = field(
        default=True,
        metadata={"help": "gpt, fuse_attention_qkv"},
    )
    recompute_granularity: str = field(
        default="full",
        metadata={"help": "full core_attn"},
    )
    virtual_pp_degree: int = field(
        default=1,
        metadata={"help": "virtual_pp_degree"},
    )

    continue_training: bool = field(
        default=False,
        metadata={
            "help": "Pre-training from existing paddlenlp model weights. Default Fasle and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models."
        },
    )


def create_pretrained_dataset(
    data_args,
    training_args,
    data_file,
    tokenizer,
):

    train_valid_test_num_samples = [
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps,
        training_args.per_device_eval_batch_size
        * training_args.dataset_world_size
        * training_args.eval_iters
        * (training_args.max_steps // training_args.eval_steps + 1),
        training_args.per_device_eval_batch_size * training_args.dataset_world_size * training_args.test_iters,
    ]

    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_valid_test_num_samples[0]))
    print_rank_0("    validation: {}".format(train_valid_test_num_samples[1]))
    print_rank_0("    test:       {}".format(train_valid_test_num_samples[2]))

    # Build the datasets.
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=data_file,
        data_impl="mmap",
        splits_string=data_args.split,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=data_args.max_seq_length,
        seed=training_args.seed,
        skip_warmup=False,
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode")
        # input_ids, loss_mask, attention_mask, position_ids, labels = data
        input_ids = data["text"]
        logger.info(tokenizer._decode(input_ids))
        # logger.info(tokenizer._decode(labels))
        # logger.info(tokenizer.convert_ids_to_tokens(input_ids))

    # eod_token = tokenizer.eos_token_id
    from paddlenlp.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        tokens_ = stack_fn(x["text"] for x in data)
        # Unpack.
        tokens_ = paddle.to_tensor(tokens_, dtype="int64")
        labels = tokens_[:, 1:]
        tokens = tokens_[:, :-1]

        # Loss mask.
        # loss_mask = paddle.ones(tokens.shape, dtype=paddle.float32)
        # loss_mask[data == eod_token] = 0.0

        return {
            "input_ids": tokens,
            # "token_type_ids": out[1],
            # "attention_mask": out[2],
            # "loss_mask": loss_mask,
            "labels": labels,
        }

    print_dataset(train_dataset[0])
    print_dataset(valid_dataset[0])
    print_dataset(test_dataset[0])

    return train_dataset, valid_dataset, test_dataset, _collate_data


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
        # keep eval_dataloader
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
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
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

    config_class, model_class = MODEL_CLASSES[model_args.model_type]

    tokenizer = Ernie35Tokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    config = config_class.from_pretrained(model_args.model_name_or_path)
    config.max_position_embeddings = max(config.max_position_embeddings, data_args.max_seq_length)
    if not model_args.continue_training:
        config.vocab_size = max(config.vocab_size, ((tokenizer.vocab_size - 1) // 128 + 1) * 128)
        logger.info(f"Reset vocab size to {config.vocab_size} for batter amp peformance.")

    config.use_flash_attention = model_args.use_flash_attention
    config.fuse_ln = model_args.use_fused_ln
    config.fuse_attention_qkv = model_args.fuse_attention_qkv
    config.recompute_granularity = model_args.recompute_granularity
    config.virtual_pp_degree = model_args.virtual_pp_degree
    config.use_recompute = training_args.recompute

    config.tensor_parallel_degree = training_args.tensor_parallel_degree
    config.tensor_parallel_rank = training_args.tensor_parallel_rank

    print("Final pre-training config:", config)

    # Set the dtype for loading model
    dtype = "float32"
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    if model_args.continue_training:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            dtype=dtype,
            use_progressive_seq_len=True,
        )
    else:
        model = model_class.from_config(config, dtype=dtype)

    # Create the learning_rate sheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps
    if training_args.warmup_steps > 0:
        warmup_steps = training_args.warmup_steps
    else:
        warmup_steps = training_args.warmup_ratio * training_args.max_steps

    lr_scheduler = None
    if training_args.lr_scheduler_type.value == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    elif training_args.lr_scheduler_type.value == "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )

    data_file = get_train_data_file(data_args)
    train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(
        data_args, training_args, data_file, tokenizer
    )

    trainer = PretrainingTrainer(
        model=model,
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
