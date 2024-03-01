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
import copy
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import paddle

from paddlenlp.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
    print_rank_0,
)
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from paddlenlp.trainer.trainer_utils import IntervalStrategy
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    register_sequence_parallel_allreduce_hooks,
)
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger
from paddlenlp.utils.transformer_engine_utils import TransformerEngineHelper


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
    # NOTE(gongenlei): new add autotuner_benchmark
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )
    transformer_engine_backend: str = field(
        default=None,
        metadata={"help": "gpt, whether to use transformer engine backend, [None, 'paddle', 'transformer_engine']"},
    )
    use_fp8: bool = field(
        default=False,
        metadata={"help": "gpt, whether to use fp8 training"},
    )

    def __post_init__(self):
        super().__post_init__()
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

    model_name_or_path: str = field(
        default="__internal_testing__/tiny-random-llama",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"},
    )
    use_fused_rms_norm: bool = field(
        default=False,
        metadata={"help": "llama or other model, use_fused_rms_norm"},
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
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "The hidden dropout prob."})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"help": "The attention hidden dropout prob."})

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
    te_init_weight_path: str = field(
        default=None,
        metadata={"help": "path to initial weights for TE"},
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
        logger.info(tokenizer._decode(list(input_ids)))

    from paddlenlp.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        tokens_ = stack_fn([x["text"] for x in data])

        labels = copy.deepcopy(tokens_)[:, 1:]
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
        files = [x.replace(".idx", "") for x in files]

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret

    return files


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


class NvtxCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self._nodeid = int(os.getenv("SLURM_NODEID", default="0"))
        self._enable_profile = int(os.environ.get("ENABLE_PROFILE", 0))
        self._start_step = int(os.environ.get("PROFILE_START_STEP", 0))
        self._stop_step = int(os.environ.get("PROFILE_STOP_STEP", 0))
        self._emit_nvtx = int(os.environ.get("PROFILE_EMIT_NVTX", 0))
        profile_node = int(os.environ.get("PROFILE_NODEID", 0))
        if self._nodeid != profile_node:
            self._enable_profile = 0
        if self._enable_profile and self._emit_nvtx:
            paddle.fluid.core.nvprof_enable_record_event()

    def on_step_begin(self, args, state, control, logs=None, inputs=None, timer=None, **kwargs):
        if self._enable_profile and state.global_step == self._start_step:
            paddle.fluid.core.nvprof_start()
        if self._enable_profile and (state.global_step == (self._stop_step + 1)):
            paddle.fluid.core.nvprof_stop()


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments))
    # Support format as "args.json --arg1 value1 --arg2 value2.”
    # In case of conflict, command line arguments take precedence.
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.enable_linear_fused_grad_add:
        from fused_layers import mock_layers

        mock_layers()

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    if data_args.data_cache is not None:
        os.makedirs(data_args.data_cache, exist_ok=True)

    paddle.set_device(training_args.device)
    set_seed(seed=training_args.seed)

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

    if training_args.transformer_engine_backend is not None:
        assert training_args.transformer_engine_backend in [
            "paddle",
            "transformer_engine",
        ], "Only support paddle and transformer_engine backend"
    config.transformer_engine_backend = training_args.transformer_engine_backend
    config.use_fp8 = training_args.use_fp8

    # Config for model using dropout, such as GPT.
    config.hidden_dropout_prob = model_args.hidden_dropout_prob
    config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob

    config.sep_parallel_degree = training_args.sep_parallel_degree
    if config.sequence_parallel:
        assert config.tensor_parallel_degree > 1, "tensor_parallel_degree must be larger than 1 for sequence parallel."
    assert (
        config.num_attention_heads % config.sep_parallel_degree == 0
    ), f"num_attention_heads:{config.num_attention_heads} must be divisible by sep_parallel_degree {config.sep_parallel_degree}"

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
        if "LLama" in str(config.architectures):
            try:
                from register_reshard import register_pp_reshard_information

                register_pp_reshard_information(config.num_hidden_layers)
            except:
                print("Not register llama pp reshard information.")

    if model_args.continue_training:
        # NOTE(gongenlei): new add
        if training_args.autotuner_benchmark:
            model = model_class.from_config(config, dtype=dtype)
        else:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                dtype=dtype,
            )
    else:
        model = model_class.from_config(config, dtype=dtype)

    if model_args.sequence_parallel:
        register_sequence_parallel_allreduce_hooks(
            model, training_args.gradient_accumulation_steps, model_args.fuse_sequence_parallel_allreduce
        )

    if training_args.recompute:
        model.recompute_enable()

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
        data_args,
        training_args,
        data_file,
        tokenizer,
        need_data=training_args.should_load_dataset,
    )

    total_effective_tokens = (
        training_args.per_device_train_batch_size
        * training_args.dataset_world_size
        * training_args.max_steps
        * training_args.gradient_accumulation_steps
        * data_args.max_seq_length
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is None and model_args.te_init_weight_path is None:
        logger.info("No checkpoint. Initializing model from scratch")
        model.init_weights()

    callbacks = [NvtxCallback()]

    trainer = PretrainingTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(None, lr_scheduler),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    if model_args.te_init_weight_path is not None:
        if checkpoint is not None:
            raise ValueError(
                "Please do not provide last_checkpoint and te_init_weight_path at the same time."
                "To load TE initial weights, please clean up the output_dir and remove --resume_from_checkpoint."
            )
        logger.info(f"Loading TE initial weights from {model_args.te_init_weight_path}")
        TransformerEngineHelper.reset_te_init_weights(trainer, model_args.te_init_weight_path)

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

    if training_args.should_load_dataset:
        effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
        print(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")
        print(f"ips: {effective_tokens_per_second:.2f} tokens/s")


if __name__ == "__main__":
    main()
