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
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler

from paddlenlp.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
    print_rank_0,
)
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.trainer.utils.reshard import NodeModelState, all_gather_state_dict
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
)
from paddlenlp.utils.log import logger


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
        * training_args.data_parallel_degree
        * training_args.max_steps
        * training_args.gradient_accumulation_steps,
        training_args.per_device_eval_batch_size
        * training_args.data_parallel_degree
        * training_args.eval_iters
        * (training_args.max_steps // training_args.eval_steps + 1),
        training_args.per_device_eval_batch_size * training_args.data_parallel_degree * training_args.test_iters,
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
    decay_parameters = [
        p.name
        for n, p in model.named_parameters()
        if (not any(nd in n for nd in ["bias", "norm"])) or "llama.norm.weight" in n
    ]

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


def get_mesh(pp_idx=0):
    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp")[pp_idx]
    return mesh


def _prepare_pipeline_inputs_func(inputs):
    first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
    last_stage_keys = ["labels"]

    def get_expected_keys(inputs, keys):
        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
        if len(ret) == 1:
            ret = ret[0]
        return ret

    if type(inputs) is dict or type(inputs) is OrderedDict:
        return [
            get_expected_keys(inputs, first_stage_keys),
            get_expected_keys(inputs, last_stage_keys),
        ]

    keys = list(inputs[0].keys())
    inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
    return [
        get_expected_keys(inputs_batch, first_stage_keys),
        get_expected_keys(inputs_batch, last_stage_keys),
    ]


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
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": training_args.data_parallel_degree,
            "mp_degree": training_args.tensor_parallel_degree,
            "pp_degree": training_args.pipeline_parallel_degree,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

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

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

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

    model_class = AutoModelForCausalLM
    if training_args.pipeline_parallel_degree > 1:
        model_class = AutoModelForCausalLMPipe

    if model_args.continue_training:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            dtype=dtype,
        )
    else:
        model = model_class.from_config(config, dtype=dtype)

    print("====type===", type(model))

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

    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)
    # skip grad sync
    load_model(model)
    assert optimizer._dp_enable
    # hack for align with auto
    # optimizer._dp_enable = False

    def loss_func(loss):
        return loss
        # hcg = fleet.get_hybrid_communicate_group()
        # group = hcg.get_data_parallel_group()
        # return LossMean.apply(loss, group)

    print_config(training_args)

    # create sampler and dataloader
    # each rank read (training_args.per_device_train_batch_size * training_args.data_parallel_degree) samples
    print(
        "dp_rank: ", dist.get_rank() // (training_args.pipeline_parallel_degree * training_args.tensor_parallel_degree)
    )
    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=False,
        num_replicas=training_args.data_parallel_degree,
        rank=dist.get_rank() // (training_args.pipeline_parallel_degree * training_args.tensor_parallel_degree),
        drop_last=training_args.dataloader_drop_last,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_train_epochs = training_args.max_steps // num_update_steps_per_epoch + int(
        training_args.max_steps % num_update_steps_per_epoch > 0
    )

    global_step = 1
    pp_data_buffer = []
    load_model(model)
    model.train()
    model._prepare_pipeline_inputs_func = _prepare_pipeline_inputs_func
    for epoch_idx in range(num_train_epochs):
        for step, inputs in enumerate(train_dataloader):
            input_ids, labels = inputs["input_ids"], inputs["labels"]
            print(f"===> input_ids:  {input_ids._md5sum()}")
            print(f"===> labels:  {labels._md5sum()}")
            pp_data_buffer.append(inputs)
            if len(pp_data_buffer) < training_args.gradient_accumulation_steps:
                continue

            pp_inputs = model._prepare_pipeline_inputs_func(pp_data_buffer)
            model.micro_batch_size = training_args.per_device_train_batch_size
            model.accumulate_steps = training_args.gradient_accumulation_steps

            pp_inputs = model._prepare_training(pp_inputs, optimizer, lr_scheduler)

            loss = model.forward_backward_pipeline(pp_inputs)
            # hack for align with auto
            # sync_grad(model)
            # print_grad(model)
            optimizer.step()
            # print_param(model)
            lr_scheduler.step()
            optimizer.clear_grad()

            print(f"global_step {global_step}; loss {loss.item()}; ls {optimizer.get_lr()}")
            pp_data_buffer.clear()

            if global_step >= 1:
                # save_model(model)
                sys.exit(0)

            global_step += 1


def save_model(model):
    hcg = fleet.get_hybrid_communicate_group()
    dp_rank = hcg.get_data_parallel_rank()
    mp_degree = hcg.get_model_parallel_world_size()
    mp_rank = hcg.get_model_parallel_rank()
    pp_rank = hcg.get_stage_id()
    if dp_rank > 0:
        return
    state_dict = model.state_dict()
    for (k, v) in state_dict.items():
        print(f"{k}=>{v.name} {v.shape}")
    paddle.save(state_dict, f"hand/pp{pp_rank:02d}mp{mp_rank:02d}.pdparams")
    group = hcg.get_model_parallel_group()

    # evenly ditribute param
    node_model_state = NodeModelState()
    node_model_state.add_weights(state_dict, mp_rank)

    def merge_func(k, v):
        assert len(v) == mp_degree
        tensor_list = [e[1] for e in v]
        return merge_mp_tensor_list(k, tensor_list)

    node_model_state = node_model_state.even_distribute(group)
    node_model_state = node_model_state.collapse_key().merge_items(merge_func)

    def filter_func(name):
        return True

    all_state_dict = all_gather_state_dict(node_model_state.model_weights, filter_func, group)
    if mp_rank > 0:
        return
    paddle.save(all_state_dict, f"hand/pp{pp_rank:02d}.pdparams")
    group = hcg.get_pipe_parallel_group()
    all_state_dict = all_gather_state_dict(all_state_dict, filter_func, group)
    if pp_rank > 0:
        return
    paddle.save(all_state_dict, "hand/all.pdparams")


def merge_tensor(tensor_list, fuse_num, axis):
    if fuse_num > 1:
        part_list = [paddle.split(e, num_or_sections=fuse_num, axis=axis) for e in tensor_list]
        fuse_list = [paddle.concat(x=e, axis=axis) for e in zip(*part_list)]
        return paddle.concat(x=fuse_list, axis=axis)
    else:
        return paddle.concat(x=tensor_list, axis=axis)


def load_model(model):
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    pp_rank = hcg.get_stage_id()
    state_dict = paddle.load(f"hand/pp{pp_rank:02d}mp{mp_rank:02d}.pdparams")
    model.set_state_dict(state_dict)


class LossMean(PyLayer):
    @staticmethod
    def forward(ctx, inp, group):
        with paddle.no_grad():
            inps = []
            paddle.distributed.all_gather(inps, inp, group=group)
            return (inps[0] + inps[1]) / 2.0

    @staticmethod
    def backward(ctx, grad):
        return grad


def sync_grad(model):
    model_state_dict = model.state_dict()
    name_mapping = {v.name: k for (k, v) in model_state_dict.items()}
    for p in model.parameters():
        assert p.name in name_mapping
        grad = p.grad
        reduce_dp(grad)


def print_grad(model):
    model_state_dict = model.state_dict()
    name_mapping = {v.name: k for (k, v) in model_state_dict.items()}
    for p in model.parameters():
        assert p.name in name_mapping
        grad = p.grad
        grad = merge_mp(name_mapping[p.name], grad)
        print(f"{name_mapping[p.name]} {p.name}_grad shape: {grad.shape} md5sum: {grad._md5sum()}")


def print_param(model):
    model_state_dict = model.state_dict()
    name_mapping = {v.name: k for (k, v) in model_state_dict.items()}
    for p in model.parameters():
        tmp = merge_mp(name_mapping[p.name], p)
        print(f"{name_mapping[p.name]} {p.name} shape: {tmp.shape} md5sum: {tmp._md5sum()}")


def merge_mp(k, input):
    hcg = fleet.get_hybrid_communicate_group()
    mp_degree = hcg.get_model_parallel_world_size()
    if mp_degree <= 1:
        return input
    else:
        group = hcg.get_model_parallel_group()
        with paddle.no_grad():
            inps = []
            paddle.distributed.all_gather(inps, input, group=group)
            return merge_mp_tensor_list(k, inps)


def concat_dp(input):
    hcg = fleet.get_hybrid_communicate_group()
    dp_degree = hcg.get_data_parallel_world_size()
    if dp_degree <= 1:
        return input
    else:
        group = hcg.get_data_parallel_group()
        return concat(input, 0, group)


def concat(input, axis, group):
    with paddle.no_grad():
        inps = []
        paddle.distributed.all_gather(inps, input, group=group)
        return paddle.concat(x=inps, axis=axis)


def reduce_dp(input):
    hcg = fleet.get_hybrid_communicate_group()
    dp_degree = hcg.get_data_parallel_world_size()
    if dp_degree <= 1:
        return input
    else:
        group = hcg.get_data_parallel_group()
        with paddle.no_grad():
            paddle.distributed.all_reduce(input, group=group)
            return input


def map_structure_name(k):
    if "_layers.llama" in k:
        return k
    hcg = fleet.get_hybrid_communicate_group()
    pp_degree = hcg.get_pipe_parallel_world_size()
    if pp_degree < 2:
        return k
    fs = k.split(".")
    idx = int(fs[1])
    if idx == 0:
        return "_layers.llama.embed_tokens.weight"
    if idx == 33:
        return "_layers.llama.norm.weight"
    if idx == 34:
        return "_layers.lm_head.weight"
    else:
        return f"_layers.llama.layers.{idx-1}." + ".".join(fs[2:])


def merge_mp_tensor_list(k, tensor_list):
    # merge by col
    k = map_structure_name(k)
    if "self_attn.qkv_proj.weight" in k:
        return merge_tensor(tensor_list, 3, 1)
    elif "self_attn.qkv_proj.bias" in k:
        return merge_tensor(tensor_list, 3, 0)
    elif "self_attn.q_proj.weight" in k:
        return merge_tensor(tensor_list, 1, 1)
    elif "self_attn.k_proj.weight" in k:
        return merge_tensor(tensor_list, 1, 1)
    elif "self_attn.v_proj.weight" in k:
        return merge_tensor(tensor_list, 1, 1)
    elif "mlp.up_gate_proj.weight" in k:
        return merge_tensor(tensor_list, 2, 1)
    elif "mlp.up_proj.weight" in k:
        return merge_tensor(tensor_list, 1, 1)
    elif "mlp.gate_proj.weight" in k:
        return merge_tensor(tensor_list, 1, 1)
    elif "lm_head.weight" in k:
        return merge_tensor(tensor_list, 1, 1)
    elif "mlp.up_gate_proj.bias" in k:
        return merge_tensor(tensor_list, 2, 0)
    # merge by row
    elif "self_attn.o_proj.weight" in k:
        return merge_tensor(tensor_list, 1, 0)
    elif "mlp.down_proj.weight" in k:
        return merge_tensor(tensor_list, 1, 0)
    elif "embed_tokens.weight" in k:
        return merge_tensor(tensor_list, 1, 0)
    else:
        assert "norm" in k, k
        # duplicate
        return tensor_list[0]


if __name__ == "__main__":
    main()
