# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
GPT/Llama pretraining scripts.
"""
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paddle
from configuration import GPTConfig
from modeling import GPTForCausalLM
from modeling_pp import GPTForCausalLMPipe

from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
    speed_metrics,
)
from paddlenlp.transformers import (
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
)
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt": (
        GPTConfig,
        GPTForCausalLM,
    ),
}

from dataset import GPTDataset, get_train_valid_test_split_


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


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
    cache_prefix: str = field(default=None, metadata={"help": "The prefix of the cached dataset."})
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

    model_type: Optional[str] = field(default="gpt", metadata={"help": "Only support for gpt pre-training for now."})
    model_name_or_path: str = field(
        default="gpt2-medium-en",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    output_attentions: bool = field(default=False, metadata={"help": "Whether output attention weights"})
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    fused_linear: bool = field(
        default=False,
        metadata={"help": "gpt, whether to fuse linear projection"},
    )
    fuse_attention_qkv: bool = field(
        default=False,
        metadata={"help": "gpt, whether to fuse attention qkv"},
    )
    enable_fuse_transformer: bool = field(
        default=False,
        metadata={"help": "gpt, enable_fuse_transformer"},
    )
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "The hidden dropout prob."})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"help": "The attention hidden dropout prob."})


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

    input_prefix = data_file[0]

    for suffix in ["_ids.npy", "_idx.npz"]:
        if not os.path.isfile(input_prefix + suffix):
            raise ValueError("File Not found, %s" % (input_prefix + suffix))

    sample_ids = np.load(input_prefix + "_ids.npy", mmap_mode="r", allow_pickle=True)
    # All documment ids, extend as 1-D array.

    process_data = np.load(input_prefix + "_idx.npz")
    # The len(sample_lens) num of docs
    # The sum(sample_lens) should equal len(sample_ids)
    sample_lens = process_data["lens"]

    splits = get_train_valid_test_split_(data_args.split, len(sample_lens))
    assert len(sample_lens) >= splits[-1], "The document nums should larger than max of splits, but %s < %s" % (
        len(sample_lens),
        splits[-1],
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode")
        input_ids, loss_mask, attention_mask, position_ids, labels = data
        logger.info(tokenizer._decode(input_ids))
        # logger.info(tokenizer._decode(labels))
        # logger.info(tokenizer.convert_ids_to_tokens(input_ids))

    def build_dataset(index, name):
        dataset = GPTDataset(
            file_prefix=os.path.join(data_args.cache_prefix, os.path.basename(input_prefix)),
            build_data_file=training_args.local_process_index == 0,
            micro_batch_size=training_args.per_device_train_batch_size
            if name == "train"
            else training_args.per_device_eval_batch_size,
            name="gpt_" + name,
            max_seq_len=data_args.max_seq_length,
            num_samples=train_valid_test_num_samples[index],
            documents=np.arange(splits[index], splits[index + 1]),
            sample_ids=sample_ids,
            sample_lens=sample_lens,
            eos_id=tokenizer.eos_token_id,
            seed=training_args.seed,
        )
        print_dataset(dataset[0], name)
        return dataset

    from paddlenlp.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        # 0:input_ids, 1:loss_mask, 2:attention_mask, 3:position_ids, 4:labels
        for i in (0, 1, 2, 3, 4):
            out[i] = stack_fn([x[i] for x in data])

        return {
            "input_ids": out[0],
            "attention_mask": out[2],
            "labels": out[4],
        }

    # Note, data should be broardcast to all devices.
    # for train, valid, test, the distinct data num is data_world_size
    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

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
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    if data_args.cache_prefix is None:
        data_args.cache_prefix = data_args.input_dir
    else:
        os.makedirs(data_args.cache_prefix, exist_ok=True)

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
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    config_class, model_class = MODEL_CLASSES[model_args.model_type]

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    config = config_class.from_pretrained(model_args.model_name_or_path)
    config.output_attentions = model_args.output_attentions
    config.max_position_embeddings = max(config.max_position_embeddings, data_args.max_seq_length)
    config.hidden_dropout_prob = model_args.hidden_dropout_prob
    config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
    config.enable_fuse_transformer = model_args.enable_fuse_transformer
    config.fuse_attention_qkv = model_args.fuse_attention_qkv
    config.use_recompute = training_args.recompute
    config.use_flash_attention = model_args.use_flash_attention
    config.lm_shift_labels = False

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

    if training_args.pipeline_parallel_degree > 1:
        model_class = GPTForCausalLMPipe

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=dtype,
        load_state_as_np=True,
    )

    # Create the learning_rate sheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps
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
    checkpoint = None

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
