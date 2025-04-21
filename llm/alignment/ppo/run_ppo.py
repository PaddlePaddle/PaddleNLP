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
import os
import sys
from functools import partial
from typing import Dict

import paddle
from paddle.distributed import fleet

from paddlenlp.datasets.rlhf_datasets import RLHFDataset, collate_fn
from paddlenlp.rl.models.score_model import AutoModelForScore
from paddlenlp.rl.trainer.ppo_trainer import PPOTrainer
from paddlenlp.rl.utils.config_utils import (
    DataArgument,
    ModelArgument,
    TrainingArguments,
)
from paddlenlp.rl.utils.offload_utils import offload_tensor_to_cpu
from paddlenlp.rl.utils.reshard_utils import init_rollout_env
from paddlenlp.rl.utils.timer_utils import timers_scope_runtimer
from paddlenlp.trainer import (
    EarlyStoppingCallback,
    PdArgumentParser,
    get_last_checkpoint,
)
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger


def process_args(model_args: ModelArgument, data_args: DataArgument, training_args: TrainingArguments):
    training_args.max_src_len = data_args.max_prompt_len
    training_args.actor_model_name_or_path = model_args.actor_model_name_or_path
    training_args.max_length = data_args.max_length

    if training_args.use_rm_server:
        if model_args.reward_server is None:
            raise ValueError("Please specify reward_server when use_rm_server is true.")
        logger.info(f"Use reward server: {model_args.reward_server} for training.")
        if training_args.rl_algorithm == "ppo" and model_args.critic_model_name_or_path is None:
            raise ValueError("Please specify critic_model_name_or_path when use_rm_server is true.")
    else:
        if model_args.reward_model_name_or_path is None:
            raise ValueError("Please specify reward_model_name_or_path when use_rm_server is false.")

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"world_size: {training_args.world_size}, " + f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    return model_args, data_args, training_args


def create_actor_models(
    model_args: ModelArgument,
    data_args: DataArgument,
    training_args: TrainingArguments,
    common_config: Dict,
):
    with timers_scope_runtimer("Actor model loading time"):
        # actor model
        actor_model_config: PretrainedConfig = AutoConfig.from_pretrained(
            model_args.actor_model_name_or_path,
            tensor_parallel_output=training_args.tensor_parallel_output,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            recompute_granularity=model_args.recompute_granularity,
            dtype=training_args.model_dtype,
            recompute=training_args.recompute,
            recompute_use_reentrant=training_args.recompute_use_reentrant,
            **common_config,
        )

        actor_model_config.use_fused_head_and_loss_fn = training_args.use_fused_head_and_loss_fn
        actor_model_config.set_attn_func = True
        actor_model_config.max_position_embeddings = data_args.max_length
        actor_model_config.use_sparse_head_and_loss_fn = False
        actor_model_config.fused_linear = model_args.fused_linear
        actor_model_config.use_fused_rms_norm = training_args.use_fused_rms_norm
        actor_model_config.seq_length = data_args.max_length
        actor_model_config.max_sequence_length = data_args.max_length
        print(f"Loading Actor model with config:\n\t{actor_model_config}\n")

        if not training_args.autotuner_benchmark:
            actor_model = AutoModelForCausalLM.from_pretrained(
                model_args.actor_model_name_or_path, config=actor_model_config
            )
        else:
            actor_model = AutoModelForCausalLM.from_config(actor_model_config)

    with timers_scope_runtimer("Actor eval model loading time"):
        if (
            training_args.rollout_tensor_parallel_degree != training_args.tensor_parallel_degree
            or training_args.pipeline_parallel_degree > 1
        ):
            actor_eval_model_config = copy.deepcopy(actor_model_config)
            actor_eval_model_config.use_fused_head_and_loss_fn = False
            with init_rollout_env(training_args.rollout_tensor_parallel_degree):
                hcg = fleet.get_hybrid_communicate_group()
                actor_eval_model_config.tensor_parallel_degree = hcg.get_model_parallel_world_size()
                actor_eval_model_config.tensor_parallel_rank = hcg.get_model_parallel_rank()
                # TODO(gongenlei): lazy load lazy guard
                actor_eval_model = AutoModelForCausalLM.from_config(actor_eval_model_config)
        else:
            actor_eval_model = None

    with timers_scope_runtimer("Reference model loading time"):
        reference_model = AutoModelForCausalLM.from_config(
            actor_model_config,
            dtype=training_args.model_dtype,
        )
        if not training_args.autotuner_benchmark:
            reference_model.set_state_dict(actor_model.state_dict())

    actor_tokenizer = AutoTokenizer.from_pretrained(
        model_args.actor_model_name_or_path,
        model_max_length=data_args.max_length,
        padding_side="left",
        tokenizer_alpha=model_args.actor_tokenizer_alpha,
        use_fast=True,
    )
    if actor_tokenizer.pad_token_id is None:
        actor_tokenizer.pad_token_id = actor_tokenizer.eos_token_id
    llm_utils.init_chat_template(actor_tokenizer, model_args.actor_model_name_or_path, model_args.chat_template)

    return actor_model, actor_eval_model, reference_model, actor_tokenizer


def create_reward_models(
    model_args: ModelArgument,
    data_args: DataArgument,
    training_args: TrainingArguments,
    common_config: Dict,
):
    with timers_scope_runtimer("Reward model loading time"):
        reward_model_config = AutoConfig.from_pretrained(
            model_args.reward_model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            dtype=training_args.model_dtype,
            recompute=training_args.critic_recompute,
            recompute_granularity=model_args.critic_recompute_granularity,
            recompute_use_reentrant=training_args.recompute_use_reentrant,
            **common_config,
        )
        reward_model_config.max_position_embeddings = data_args.max_length
        reward_model_config.use_sparse_head_and_loss_fn = False
        reward_model_config.fused_linear = model_args.fused_linear
        print(f"Loading Reward model with config:\n\t{reward_model_config}\n")

        config = copy.deepcopy(reward_model_config)
        if training_args.eval_mode is not None:
            if training_args.eval_mode == "single":
                config.tensor_parallel_degree = -1
                config.tensor_parallel_rank = 0

        if not training_args.autotuner_benchmark:
            reward_model = AutoModelForScore.from_pretrained(
                model_args.reward_model_name_or_path,
                config=config,
                score_type="reward",
                do_normalize=False,
            )
        else:
            reward_model = AutoModelForScore.from_config(
                config,
                score_type="reward",
                do_normalize=False,
            )

    reward_tokenizer = AutoTokenizer.from_pretrained(
        model_args.reward_model_name_or_path,
        model_max_length=data_args.max_length,
        padding_side="right",
        tokenizer_alpha=model_args.reward_tokenizer_alpha,
        use_fast=True,
    )
    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id
    llm_utils.init_chat_template(reward_tokenizer, model_args.reward_model_name_or_path, model_args.chat_template)
    return reward_model, reward_tokenizer


def create_critic_models(
    model_args: ModelArgument,
    data_args: DataArgument,
    training_args: TrainingArguments,
    common_config: Dict,
    reward_model,
):
    with timers_scope_runtimer("Critic model loading time"):
        reward_model_config = reward_model.config
        if model_args.critic_model_name_or_path is None:
            model_args.critic_model_name_or_path = model_args.reward_model_name_or_path
            critic_model = AutoModelForScore.from_config(
                reward_model_config,
                dtype=training_args.model_dtype,
                score_type="critic",
                do_normalize=False,
                clip_range_value=training_args.clip_range_value,
                **common_config,
            )
            if not training_args.autotuner_benchmark:
                critic_model.set_state_dict(reward_model.state_dict())
        else:
            if not training_args.autotuner_benchmark:
                critic_model = AutoModelForScore.from_pretrained(
                    model_args.critic_model_name_or_path,
                    config=reward_model_config,
                    score_type="critic",
                    do_normalize=False,
                    clip_range_value=training_args.clip_range_value,
                    **common_config,
                )
            else:
                critic_model = AutoModelForScore.from_config(
                    reward_model_config,
                    score_type="critic",
                    do_normalize=False,
                    clip_range_value=training_args.clip_range_value,
                    **common_config,
                )

    critic_tokenizer = AutoTokenizer.from_pretrained(
        model_args.critic_model_name_or_path,
        model_max_length=data_args.max_length,
        padding_side="left",
        tokenizer_alpha=model_args.reward_critic_tokenizer_alpha,
        use_fast=True,
    )
    if critic_tokenizer.pad_token_id is None:
        critic_tokenizer.pad_token_id = critic_tokenizer.eos_token_id
    llm_utils.init_chat_template(critic_tokenizer, model_args.critic_model_name_or_path, model_args.chat_template)

    if training_args.eval_mode is not None:
        config = copy.deepcopy(critic_model.config)
        if training_args.eval_mode == "single":
            config.tensor_parallel_degree = -1
            config.tensor_parallel_rank = 0
        with timers_scope_runtimer("Reward critic eval model loading time"):
            critic_eval_model = AutoModelForScore.from_config(config)
    else:
        critic_eval_model = None

    return critic_model, critic_eval_model, critic_tokenizer


def create_rl_dataset(data_args, training_args, tokenizer):
    requires_label = True if training_args.use_rm_server else False
    train_ds = RLHFDataset(
        dataset_name_or_path=data_args.train_datasets,
        tokenizer=tokenizer,
        max_prompt_len=data_args.max_prompt_len,
        requires_label=requires_label,
        label_key=data_args.label_key,
        splits="train",
    )
    dev_ds = RLHFDataset(
        dataset_name_or_path=data_args.eval_datasets,
        tokenizer=tokenizer,
        max_prompt_len=data_args.max_prompt_len,
        requires_label=requires_label,
        label_key=data_args.label_key,
        splits="dev",
    )
    return train_ds, dev_ds


def main():
    # Arguments
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # pre-precess args
    model_args, data_args, training_args = process_args(model_args, data_args, training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    common_config = dict(
        use_flash_attention=model_args.use_flash_attention,
        sequence_parallel=training_args.sequence_parallel,
        fused_rotary=False,
        max_sequence_length=data_args.max_length,
    )

    actor_model, actor_eval_model, reference_model, actor_tokenizer = create_actor_models(
        model_args, data_args, training_args, common_config
    )

    if not training_args.use_rm_server and model_args.reward_model_name_or_path is not None:
        reward_model, reward_tokenizer = create_reward_models(model_args, data_args, training_args, common_config)
    else:
        reward_model, reward_tokenizer = model_args.reward_server, actor_tokenizer

    if training_args.rl_algorithm == "ppo":
        critic_model, critic_eval_model, critic_tokenizer = create_critic_models(
            model_args, data_args, training_args, common_config, reward_model
        )
    else:
        critic_model, critic_eval_model, critic_tokenizer = None, None, None

    if training_args.should_load_dataset:
        train_ds, dev_ds = create_rl_dataset(data_args, training_args, actor_tokenizer)

    if "freeze_model" in training_args.offload_level:
        if actor_eval_model is not None:
            offload_tensor_to_cpu((actor_eval_model, "freeze_model"))
        offload_tensor_to_cpu((reference_model, "freeze_model"))

        if training_args.rl_algorithm == "ppo":
            offload_tensor_to_cpu((reward_model, "freeze_model"))
            if critic_eval_model is not None:
                offload_tensor_to_cpu((critic_eval_model, "freeze_model"))

        # NOTE(gongenlei): release memory_reserved_size to equal to memory_allocated_size
        paddle.device.cuda.empty_cache()

    def compute_metrics(eval_preds):
        accuracy = (eval_preds.predictions == 3).astype("float32").mean().item()
        return {"accuracy": accuracy}

    trainer = PPOTrainer(
        actor_model=actor_model,
        reference_model=reference_model,
        reward_model=reward_model,
        critic_model=critic_model,
        actor_model_eval=actor_eval_model,
        critic_model_eval=critic_eval_model,
        args=training_args,
        train_dataset=(train_ds if training_args.do_train and training_args.should_load_dataset else None),
        eval_dataset=(dev_ds if training_args.do_eval and training_args.should_load_dataset else None),
        actor_tokenizer=actor_tokenizer,
        reference_tokenizer=actor_tokenizer,
        reward_tokenizer=reward_tokenizer,
        critic_tokenizer=critic_tokenizer,
        data_collator=partial(
            collate_fn,
            pad_token_id=actor_tokenizer.pad_token_id,
            requires_label=True if training_args.use_rm_server else False,
        ),
        compute_metrics=compute_metrics,  # TODO: only used for grpo (kk datasets)
    )

    # TODO(gongenlei) resume_from_checkpoint is not ready
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # The early-stopping callback.
    if training_args.early_stopping:
        early_stopping_info = (
            f"Early stopping is enabled, "
            f"patience={training_args.early_stopping_patience}, "
            f"threshold={training_args.early_stopping_threshold}, "
            f"metric={training_args.metric_for_best_model}, "
            f"greater_is_better={training_args.greater_is_better}"
        )
        logger.info(early_stopping_info)
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
                early_stopping_threshold=training_args.early_stopping_threshold,
            )
        )

    # if training_args.hidden_dropout_prob or training_args.attention_probs_dropout_prob:
    #     trainer.add_callback(LayerwiseDropoutCallback())

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if not training_args.autotuner_benchmark:
            with timers_scope_runtimer("Model saving time"):
                trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
                if paddle.distributed.get_world_size() > 1:
                    paddle.distributed.barrier()

            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        # NOTE(gongenlei): set combined=False to avoid overwriting errors on AFS
        trainer.save_metrics("eval", eval_result, combined=False)


if __name__ == "__main__":
    main()
