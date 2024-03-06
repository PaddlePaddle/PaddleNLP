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
from paddle.distributed import fleet

from paddlenlp.ops import Topology
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.trainer.auto_trainer import AutoTrainer
from paddlenlp.trainer.trainer_utils import IntervalStrategy, _get_distributed_seeds
from paddlenlp.transformers import (
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    LlamaConfig,
    LlamaForCausalLM3DAuto,
    LlamaPretrainingCriterion3DAuto,
)
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "llama": (LlamaConfig, LlamaForCausalLM3DAuto, LlamaPretrainingCriterion3DAuto),
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
    fused_linear_param_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused_linear_param_grad pass, which should replace add_n_op with add_op for gradients accumulation."
        },
    )
    job_schedule_profiler_start: int = field(
        default=-1,
        metadata={"help": "The step to start job_schedule_profiler."},
    )
    job_schedule_profiler_end: int = field(
        default=-1,
        metadata={"help": "The step to end job_schedule_profiler."},
    )
    pipeline_schedule_mode: str = field(
        default="1F1B", metadata={"help": "The pipeline schedule mode, support FThenB, 1F1B, VPP and Eager-1F1B."}
    )
    sr: Optional[int] = field(default=0, metadata={"help": "The count of chunks without recompute."})
    refined_ops_patterns: Optional[List[str]] = field(
        default=None, metadata={"help": "The pattern of refined recompute."}
    )
    virtual_pipeline_seg_method: str = field(
        default="LlamaDecoderLayerAuto", metadata={"help": "The seg method of spliting pp layer for virtual pipeline."}
    )
    # NOTE(gongenlei): new add autotuner_benchmark
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.enable_auto_parallel

        # NOTE(gongenlei): new add autotuner_benchmark
        if self.autotuner_benchmark:
            self.max_steps = 5
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO

        if self.fused_linear_param_grad_add:
            fused_passes = self.strategy.fused_passes
            fused_passes.enable = True
            fused_passes.fused_passes_list.append("fused_linear_param_grad_add_pass")

        logger.info(self.strategy)


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
    vocab_size: Optional[int] = field(
        default=None,
        metadata={
            "help": ".Vocabulary size of the Llama model. Defines the number of different tokens that can be represented by the `inputs_ids`"
        },
    )
    hidden_size: Optional[int] = field(default=None, metadata={"help": "Dimension of the hidden representations."})
    intermediate_size: Optional[int] = field(default=None, metadata={"help": "Dimension of the MLP representations."})
    num_hidden_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of hidden layers in the Transformer encoder."}
    )
    num_attention_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Number of attention heads for each attention layer in the Transformer encoder."},
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


class PretrainingTrainer(AutoTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _wrap_for_dist_loader(self, train_dataloader):
        dist_loader = super()._wrap_for_dist_loader(train_dataloader)
        dist_loader._input_keys = ["input_ids", "labels"]
        return dist_loader


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
    else:
        assert not args.use_hybrid_parallel and args.enable_auto_parallel
        if dist.get_world_size() > 1:
            topo = Topology(
                dist.get_rank(),
                dist.get_world_size(),
                dp_degree=args.data_parallel_degree,
                pp_degree=args.pipeline_parallel_degree,
                mp_degree=args.tensor_parallel_degree,
                sharding_degree=1,  # auto_parallel's sharding is not orthogonal with dp, mp and pp
            )

            global_seed, local_seed, random_seed = _get_distributed_seeds(args.seed, topo)

            paddle.seed(local_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            logger.info(
                "The global seed is set to {}, local seed is set to {} and "
                "random seed is set to {}.".format(global_seed, local_seed, random_seed)
            )
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            paddle.seed(args.seed)


def get_mesh(pp_idx=0):
    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp")[pp_idx]
    return mesh


def shard_fn(layer, mesh_idx, placements):
    paran_name = layer.weight.name
    layer.weight = dist.shard_tensor(layer.weight, get_mesh(mesh_idx), placements)
    layer.weight.name = paran_name


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

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    config_class, model_class, criterion_class = MODEL_CLASSES[model_args.model_type]

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

    config.vocab_size = model_args.vocab_size if model_args.vocab_size is not None else config.vocab_size
    config.hidden_size = model_args.hidden_size if model_args.hidden_size is not None else config.hidden_size
    config.intermediate_size = (
        model_args.intermediate_size if model_args.intermediate_size is not None else config.intermediate_size
    )
    config.num_hidden_layers = (
        model_args.num_hidden_layers if model_args.num_hidden_layers is not None else config.num_hidden_layers
    )
    config.num_attention_heads = (
        model_args.num_attention_heads if model_args.num_attention_heads is not None else config.num_attention_heads
    )

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

    if training_args.strategy.pipeline.enable and config.virtual_pp_degree > 1:
        pipeline = training_args.strategy.pipeline
        pipeline.vpp_degree = config.virtual_pp_degree
        pipeline.vpp_seg_method = training_args.virtual_pipeline_seg_method

    print("Final pre-training config:", config)

    # # Set the dtype for loading model
    # dtype = "float32"
    # if training_args.fp16_opt_level == "O2":
    #     if training_args.fp16:
    #         dtype = "float16"
    #     if training_args.bf16:
    #         dtype = "bfloat16"

    with paddle.LazyGuard():
        model = model_class.from_config(config, dtype="float32")
        criterion = criterion_class(config)

    for param in model.parameters():
        assert not param._is_initialized()
        param.initialize()

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
    train_dataset, eval_dataset, test_dataset, data_collator = create_pretrained_dataset(
        data_args,
        training_args,
        data_file,
        tokenizer,
        need_data=training_args.should_load_dataset,
    )

    trainer = PretrainingTrainer(
        model=model,
        criterion=criterion,
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

        # NOTE(gongenlei): new add
        if not training_args.autotuner_benchmark:
            metrics = train_result.metrics
            if not int(os.getenv("test_ci_no_save_model", 0)):
                trainer.save_model()
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    if training_args.do_predict:
        test_ret = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_ret.metrics)

    # if training_args.should_load_dataset:
    #     effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
    #     print(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")
    #     print(f"ips: {effective_tokens_per_second:.2f} tokens/s")


def shard_model(model):
    pp_stage = 0
    for name, layer in model.named_sublayers(include_self=False):
        if hasattr(layer, "ipp"):
            pp_stage = layer.ipp
        # print(f"name {name},pp_stage {pp_stage}==>", type(layer))
        if "embed_tokens" in name:
            # embedding only support column split now. it will update in the future
            shard_fn(layer, 0, [dist.Replicate(), dist.Shard(1)])
        for n in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.qkv_proj",
            "gate_proj",
            "up_proj",
            "gate_up_fused_proj",
        ]:
            if n in name:
                shard_fn(layer, pp_stage, [dist.Replicate(), dist.Shard(1)])
                break
        for n in ["self_attn.o_proj", "down_proj"]:
            if n in name:
                shard_fn(layer, pp_stage, [dist.Replicate(), dist.Shard(0)])
                break
        if "lm_head" in name:
            shard_fn(layer, -1, [dist.Replicate(), dist.Shard(1)])


def load_model(model):
    model_state_dict = model.state_dict()
    state_dict = paddle.load("hand/all.pdparams")
    tmp = OrderedDict()
    (tmp, state_dict) = (state_dict, tmp)
    for (k, v) in tmp.items():
        k = map_structure_name(k)
        state_dict[k] = v
    model.set_state_dict(state_dict)
    assert len(model_state_dict) == len(state_dict), f"{len(model_state_dict)} vs {len(state_dict)}"
    """
    print("=======model_state_dict=======")
    for (k,v) in model_state_dict.items():
        print(f"{k}=>{v.shape}")
    """
    print("=======state_dict=======")
    for (k, v) in state_dict.items():
        assert k in model_state_dict
        print(f"{k}=>{v.shape}")


def print_grad(model):
    model_state_dict = model.state_dict()
    name_mapping = {v.name: k for (k, v) in model_state_dict.items()}
    for p in model.parameters():
        assert p.name in name_mapping
        if p.grad is not None:
            print(f"{name_mapping[p.name]} {p.name}_grad shape: {p.grad.shape} md5sum: {p.grad._md5sum()}")


def print_param(model):
    model_state_dict = model.state_dict()
    name_mapping = {v.name: k for (k, v) in model_state_dict.items()}
    for p in model.parameters():
        assert p.name in name_mapping
        if p.grad is not None:
            print(f"{name_mapping[p.name]} {p.name} shape: {p.shape} md5sum: {p._md5sum()}")


def map_structure_name(k):
    fs = k.split(".")
    idx = int(fs[1])
    if idx == 0:
        return "llama.embed_tokens.weight"
    if idx == 33:
        return "llama.norm.weight"
    if idx == 34:
        return "lm_head.weight"
    else:
        return f"llama.layers.{idx-1}." + ".".join(fs[2:])


if __name__ == "__main__":
    main()
