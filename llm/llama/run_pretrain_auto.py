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
GPT/Llama auto parallel pretraining scripts.
"""
import os
import random
import sys
import time
import types
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed.auto_parallel as auto

from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    speed_metrics,
)
from paddlenlp.transformers import (
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    LlamaConfig,
    LlamaForCausalLMAuto,
)
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "llama": (
        LlamaConfig,
        LlamaForCausalLMAuto,
    ),
}


from paddlenlp.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
    print_rank_0,
)


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
    enable_linear_fused_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused linear grad add strategy, which will reduce elementwise add for grad accumulation in the backward of nn.Linear ."
        },
    )
    parallel_mode: str = field(default="hybrid", metadata={"help": ""})


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

    data_impl: str = field(default="mmap", metadata={"help": "The format of the preprocessed data."})
    skip_warmup: bool = field(
        default=True,
        metadata={"help": "Whether to skip the warmup process of mmap files."},
    )
    data_cache: str = field(default=None, metadata={"help": "The path of the cached dataset."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to pre-train from.
    """

    model_type: Optional[str] = field(
        default="llama", metadata={"help": "Only support for llama pre-training for now."}
    )
    model_name_or_path: str = field(
        default="__internal_testing__/tiny-random-llama",
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
    use_fused_rms_norm: bool = field(
        default=False,
        metadata={"help": "llama, use_fused_rms_norm"},
    )
    fuse_attention_qkv: bool = field(
        default=False,
        metadata={"help": "whether to fuse attention qkv"},
    )
    fuse_attention_ffn: bool = field(
        default=False,
        metadata={"help": "whether to fuse first up and gate proj in mlp block"},
    )
    recompute_granularity: str = field(
        default="full",
        metadata={"help": "Choose among ['full', 'core_attn', 'full_attn']"},
    )
    virtual_pp_degree: int = field(
        default=1,
        metadata={"help": "virtual_pp_degree"},
    )
    continue_training: bool = field(
        default=False,
        metadata={
            "help": "Pre-training from existing paddlenlp model weights. Default False and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models."
        },
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "whether to use sequence parallel"},
    )
    fuse_sequence_parallel_allreduce: bool = field(
        default=False,
        metadata={"help": "whether to use fuse sequence parallel allreduce"},
    )
    use_fused_rope: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable rope fusion or not."},
    )
    no_recompute_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Specify the full transformer layers that should not be recomputed."},
    )
    pp_recompute_interval: int = field(
        default=1,
        metadata={
            "help": "The interval for the number of layers at which recomputation occurs. A value of 0 indicates no recomputation. Default is 0."
        },
    )
    recompute_use_reentrant: bool = field(
        default=False,
        metadata={"help": "recompute_use_reentrant"},
    )


def create_pretrained_dataset(
    data_args,
    training_args,
    data_file,
    tokenizer,
    need_data=True,
):

    check_data_split(data_args.split, training_args.do_train, training_args.do_eval, training_args.do_predict)

    train_val_test_num_samples = [
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
    if training_args.do_train:
        print_rank_0("    train:      {}".format(train_val_test_num_samples[0]))
    if training_args.do_eval:
        print_rank_0("    validation: {}".format(train_val_test_num_samples[1]))
    if training_args.do_predict:
        print_rank_0("    test:       {}".format(train_val_test_num_samples[2]))

    # Build the datasets.
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=data_file,
        data_impl=data_args.data_impl,
        splits_string=data_args.split,
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=data_args.max_seq_length,
        seed=training_args.seed,
        skip_warmup=data_args.skip_warmup,
        share_folder=data_args.share_folder,
        data_cache_path=data_args.data_cache,
        need_data=need_data,
    )

    def print_dataset(data, mode="train"):
        logger.info(f"Sample data for {mode} mode.")
        # input_ids, loss_mask, attention_mask, position_ids, labels = data
        input_ids = data["text"]

        logger.info(tokenizer._decode(input_ids))

    from paddlenlp.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        tokens_ = stack_fn([x["text"] for x in data])

        labels = tokens_[:, 1:]
        tokens = tokens_[:, :-1]

        return {
            "input_ids": tokens,
            "labels": labels,
        }

    if need_data:
        if training_args.do_train:
            print_dataset(train_dataset[0], "train")
        if training_args.do_eval:
            print_dataset(valid_dataset[0], "valid")
        if training_args.do_predict:
            print_dataset(test_dataset[0], "test")

    return train_dataset, valid_dataset, test_dataset, _collate_data


def get_train_data_file(args):
    if len(args.input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return args.input_dir.split()
    else:
        files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and ("_idx.npz" in str(f) or ".idx" in str(f)))
        ]
        files = [x.replace("_idx.npz", "") for x in files]
        files = [x.replace(".idx", "") for x in files]  # add

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret

    return files


def create_optimizer(model, lr_scheduler, training_args):
    decay_parameters = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]

    def apply_decay_param_fun(x):
        return x in decay_parameters

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optimizer_cls(
        learning_rate=lr_scheduler if lr_scheduler is None else lr_scheduler,
        apply_decay_param_fun=apply_decay_param_fun,
        parameters=model.parameters(),
        weight_decay=training_args.weight_decay,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(training_args.max_grad_norm)
        if training_args.max_grad_norm > 0
        else None,
        **optimizer_kwargs,
    )

    return optimizer


def print_config(args, key=""):
    """
    print config values
    """
    logger.info("=" * 60)
    if args is None:
        args = args
        key = "Training"
    import paddlenlp

    logger.info("{:^40}".format("{} Configuration Arguments".format(key)))
    logger.info("{:30}: {}".format("paddle commit id", paddle.version.commit))
    logger.info("{:30}: {}".format("paddlenlp commit id", paddlenlp.version.commit))

    for a in dir(args):
        if a[:2] != "__":  # don't print double underscore methods
            v = getattr(args, a)
            if not isinstance(v, types.MethodType):
                logger.info("{:30}: {}".format(a, v))

    logger.info("")


def init_seed(seed: int = 1234, args=None):
    if args is None:
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    if args is not None:
        if args.use_hybrid_parallel:
            from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

            random.seed(args.seed + args.dataset_rank)
            np.random.seed(args.seed + args.dataset_rank)
            paddle.seed(args.seed + args.dataset_rank)

            # local_seed/ global_seed is used to control dropout in ModelParallel
            local_seed = args.seed + 59999 + args.tensor_parallel_rank * 10 + args.pipeline_parallel_rank * 1000
            global_seed = args.seed + 100003 + args.dataset_rank
            tracker = get_rng_state_tracker()

            if "global_seed" not in tracker.states_:
                tracker.add("global_seed", global_seed)
            if "local_seed" not in tracker.states_:
                tracker.add("local_seed", local_seed)
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            paddle.seed(args.seed)


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.enable_linear_fused_grad_add:
        from fused_layers import mock_layers

        mock_layers()

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    if data_args.data_cache is not None:
        os.makedirs(data_args.data_cache, exist_ok=True)

    init_seed(args=training_args)
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

    config_class, model_class = MODEL_CLASSES[model_args.model_type]

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    config = config_class.from_pretrained(model_args.model_name_or_path)

    config.seq_length = data_args.max_seq_length
    # There are some technique extend RotaryEmbedding context. so don't change max_position_embeddings
    if not model_args.continue_training:
        config.max_position_embeddings = max(config.max_position_embeddings, data_args.max_seq_length)

    if not model_args.continue_training:
        config.vocab_size = max(config.vocab_size, ((tokenizer.vocab_size - 1) // 128 + 1) * 128)
        logger.info(f"Reset vocab size to {config.vocab_size} for batter amp peformance.")

    if model_args.no_recompute_layers is not None:
        model_args.no_recompute_layers.sort()

    config.use_flash_attention = model_args.use_flash_attention
    config.use_fused_rms_norm = model_args.use_fused_rms_norm
    config.fuse_attention_qkv = model_args.fuse_attention_qkv
    config.fuse_attention_ffn = model_args.fuse_attention_ffn
    config.recompute_granularity = model_args.recompute_granularity
    config.virtual_pp_degree = model_args.virtual_pp_degree
    config.sequence_parallel = model_args.sequence_parallel
    config.fuse_sequence_parallel_allreduce = model_args.fuse_sequence_parallel_allreduce
    config.use_fused_rope = model_args.use_fused_rope
    config.no_recompute_layers = model_args.no_recompute_layers
    config.pp_recompute_interval = model_args.pp_recompute_interval
    config.recompute_use_reentrant = model_args.recompute_use_reentrant

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

    model = model_class._from_config(config, dtype=dtype)

    if training_args.recompute:

        def fn(layer):
            if hasattr(layer, "enable_recompute") and (layer.enable_recompute is False or layer.enable_recompute == 0):
                layer.enable_recompute = True

        model.apply(fn)

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
    train_dataset, _, _, data_collator = create_pretrained_dataset(
        data_args,
        training_args,
        data_file,
        tokenizer,
        need_data=training_args.should_load_dataset,
    )

    optimizer = create_optimizer(model, lr_scheduler, training_args)

    def loss_func(loss, outputs):
        return loss

    global_batch_size = training_args.per_device_train_batch_size * training_args.dataset_world_size
    print_config(training_args)

    engine = auto.Engine(model, loss_func, optimizer, strategy=training_args.strategy)
    engine.prepare(
        [
            paddle.static.InputSpec(
                shape=[global_batch_size, data_args.max_seq_length], dtype="int64", name="input_ids"
            ),
            paddle.static.InputSpec(shape=[global_batch_size, data_args.max_seq_length], dtype="int64", name="labels"),
        ],
        mode="train",
    )

    train_dataloader = engine.dataloader(
        dataset=train_dataset,
        batch_size=global_batch_size,
        steps_per_epoch=training_args.max_steps,
        epochs=training_args.num_train_epochs,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers,
        mode="train",
    )

    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_train_epochs = training_args.max_steps // num_update_steps_per_epoch + int(
        training_args.max_steps % num_update_steps_per_epoch > 0
    )
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.dataset_world_size
    )

    global_step = 0
    global_step_last_logged = 0
    start_time_last_logged = time.time()
    tr_loss = float(0)
    for epoch_idx in range(num_train_epochs):
        for step, inputs in enumerate(train_dataloader):
            outs = engine.run(inputs, mode="train")

            if "loss" in outs:
                tr_loss_step = np.sum(outs["loss"])
            else:
                tr_loss_step = float(0)

            if training_args.gradient_accumulation_steps > 1:
                tr_loss_step /= training_args.gradient_accumulation_steps

            tr_loss += tr_loss_step

            if lr_scheduler is not None:
                engine.optimizer._learning_rate.step()

            global_step += 1
            if (step + 1) % training_args.logging_steps == 0:
                num_steps = global_step - global_step_last_logged
                logs = {}
                logs["loss"] = round(tr_loss / num_steps, 8)
                logs["learning_rate"] = float("{0:.3e}".format(engine.optimizer.get_lr()))
                logs["global_step"] = int(global_step)
                logs.update(
                    speed_metrics(
                        split="interval",
                        start_time=start_time_last_logged,
                        num_samples=total_train_batch_size * num_steps,
                        num_steps=num_steps,
                    )
                )
                logger.info(", ".join(f"{k}: {v}" for k, v in logs.items()))

                global_step_last_logged = global_step
                start_time_last_logged = time.time()
                tr_loss = float(0)

            if step >= training_args.max_steps:
                break


if __name__ == "__main__":
    main()
