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
import types
from functools import partial

import paddle
from comm_utils import offload_tensor_to_cpu
from data import PromptOnlyDataset, SupervisedDataset
from models.score_model import LlamaModelForScore  # noqa
from ppo_trainer import PPOTrainer
from trainer_utils import DataArgument, ModelArgument, TrainingArguments

from paddlenlp.trainer import PdArgumentParser, RuntimeTimer, get_last_checkpoint
from paddlenlp.trainer.trainer_utils import ShardingOption
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger


def main():
    """
    主函数，用于运行训练。

    Args:
        无参数。

    Returns:
        None: 该函数没有返回值。

    Raises:
        无异常抛出。
    """
    # Arguments
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.max_src_len = data_args.max_prompt_len
    training_args.actor_model_name_or_path = model_args.actor_model_name_or_path

    if training_args.sequence_parallel:
        if training_args.tensor_parallel_degree <= 1:
            training_args.sequence_parallel = False
            logger.info("Tensor_parallel_degree = 1. Set sequence_parallel to False.")

    if training_args.tensor_parallel_degree <= 1:
        training_args.tensor_parallel_output = False
        logger.info("Tensor_parallel_degree = 1. Set tensor_parallel_output to False.")

    if training_args.sharding_parallel_degree > 1:
        if (
            ShardingOption.SHARD_GRAD_OP in training_args.sharding
            or ShardingOption.FULL_SHARD in training_args.sharding
        ):
            if training_args.release_grads is True:
                training_args.release_grads = False

    if training_args.unified_checkpoint and "async_save" in training_args.unified_checkpoint_config:
        training_args.unified_checkpoint_config.remove("async_save")
        logger.warning(
            "PPO training currently does not support asynchronous saving! "
            "Remove `async_save` from unified_checkpoint_config."
        )

    training_args.offload_level = training_args.offload_level.split()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    runtime_timer = RuntimeTimer("Training")

    if training_args.eval_mode is not None and len(training_args.eval_mode) == 0:
        training_args.eval_mode = None
    # if training_args.eval_mode is None and training_args.offload_level is not None:
    #     training_args.offload_level = training_args.offload_level.replace("eval", "")

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"world_size: {training_args.world_size}, " + f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
        #     raise ValueError(
        #         f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        #         "Use --overwrite_output_dir to overcome."
        #     )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load model
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    training_args.max_length = data_args.max_length

    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    if training_args.use_rm_server:
        if model_args.reward_server is None:
            raise ValueError("Please specify reward_server when use_rm_server is true.")
        logger.info(f"Use reward server: {model_args.reward_server} for training.")
        if training_args.rl_algorithm == "ppo" and model_args.reward_critic_model_name_or_path is None:
            raise ValueError("Please specify reward_critic_model_name_or_path when use_rm_server is true.")
    else:
        if model_args.reward_model_name_or_path is None:
            raise ValueError("Please specify reward_model_name_or_path when use_rm_server is false.")

    if training_args.rl_algorithm != "ppo" and training_args.use_fused_head_and_loss_fn:
        logger.warning(
            f"Fused_head_and_loss_fn currently does not support {training_args.rl_algorithm}. "
            "Reset `use_fused_head_and_loss_fn` to False."
        )
        training_args.use_fused_head_and_loss_fn = False

    model_class_lm, model_class_score = AutoModelForCausalLM, LlamaModelForScore
    if training_args.pipeline_parallel_degree > 1:
        from models.model_pp import LlamaPolicyPipe, LlamaValuePipe

        model_class_lm = LlamaPolicyPipe
        model_class_score = LlamaValuePipe
        extra_args = {
            "ptx_coeff": training_args.ptx_coeff,
            "clip_range_ratio": training_args.clip_range_ratio,
        }
    else:
        # non-pipe modelForCausalLM does not accept extra_args and use other ways
        # (StepTrainer.create_criterion) to set hyper-parameters
        extra_args = {}

    common_config = dict(
        use_flash_attention=model_args.use_flash_attention,
        sequence_parallel=training_args.sequence_parallel,
        fused_rotary=False,
        max_sequence_length=data_args.max_length,
    )

    runtime_timer.start("Actor model loading time")

    # actor model
    actor_model_config = AutoConfig.from_pretrained(
        model_args.actor_model_name_or_path,
        tensor_parallel_output=training_args.tensor_parallel_output,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        recompute_granularity=model_args.recompute_granularity,
        dtype=dtype,
        recompute=training_args.recompute,
        recompute_use_reentrant=training_args.recompute_use_reentrant,
        **common_config,
    )

    actor_model_config.use_fused_head_and_loss_fn = training_args.use_fused_head_and_loss_fn
    actor_model_config.set_attn_func = True
    actor_model_config.max_position_embeddings = data_args.max_length
    actor_model_config.use_sparse_head_and_loss_fn = False
    actor_model_config.fused_linear = model_args.fused_linear
    print(f"Loading Actor model with config:\n\t{actor_model_config}\n")

    if not training_args.autotuner_benchmark:
        actor_model = model_class_lm.from_pretrained(
            model_args.actor_model_name_or_path,
            config=actor_model_config,
            **extra_args,
            # ptx_coeff=training_args.ptx_coeff,
            # clip_range_ratio=training_args.clip_range_ratio,
        )
    else:
        actor_model = model_class_lm.from_config(
            actor_model_config,
            **extra_args,
            # ptx_coeff=training_args.ptx_coeff,
            # clip_range_ratio=training_args.clip_range_ratio,
        )

    logger.info(f"{runtime_timer.log()}")

    if training_args.eval_mode is not None:
        config = copy.deepcopy(actor_model.config)
        config.use_fused_head_and_loss_fn = False
        if training_args.eval_mode == "single":
            config.tensor_parallel_degree = -1
            config.tensor_parallel_rank = 0
        runtime_timer.start("Actor eval model loading time")
        actor_eval_model = AutoModelForCausalLM.from_config(config)
        logger.info(f"{runtime_timer.log()}")
        # TODO(guosheng): AutoModel (in `_get_model_class_from_config`) pop out
        # architecture which is necessary for infer predictor currently
        config.architectures = actor_model.config.architectures
        # actor_eval_model = AutoModelForCausalLM.from_pretrained(model_args.actor_model_name_or_path, config=config)
        # cleanup_tensor_space(actor_eval_model.state_dict())
    else:
        actor_eval_model = None

    runtime_timer.start("Actor reference model loading time")
    # todo reference model
    if training_args.eval_mode is not None:
        config = copy.deepcopy(actor_model_config)
        config.use_fused_head_and_loss_fn = False
        if training_args.eval_mode == "single":
            config.tensor_parallel_degree = -1
            config.tensor_parallel_rank = 0
        if not training_args.autotuner_benchmark:
            actor_reference_model = AutoModelForCausalLM.from_pretrained(
                model_args.actor_model_name_or_path,
                config=config,
            )
        else:
            actor_reference_model = AutoModelForCausalLM.from_config(
                config,
                dtype=dtype,
            )
    else:
        actor_reference_model = model_class_lm.from_config(
            actor_model_config,
            dtype=dtype,
        )
        if not training_args.autotuner_benchmark:
            actor_reference_model.set_state_dict(actor_model.state_dict())
    logger.info(f"{runtime_timer.log()}")

    actor_tokenizer = AutoTokenizer.from_pretrained(
        model_args.actor_model_name_or_path,
        model_max_length=data_args.max_length,
        padding_side="left",
        tokenizer_alpha=model_args.actor_tokenizer_alpha,
    )
    llm_utils.init_chat_template(actor_tokenizer, model_args.actor_model_name_or_path, model_args.chat_template)

    training_args.autotuner_benchmark = True
    if not training_args.use_rm_server and model_args.reward_model_name_or_path is not None:
        runtime_timer.start("Reward model loading time")
        # reward model
        reward_model_config = AutoConfig.from_pretrained(
            model_args.reward_model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            dtype=dtype,
            recompute=training_args.critic_recompute,
            recompute_granularity=model_args.critic_recompute_granularity,
            recompute_use_reentrant=training_args.recompute_use_reentrant,
            **common_config,
        )
        reward_model_config.num_hidden_layers = 2
        reward_model_config.max_position_embeddings = data_args.max_length
        reward_model_config.use_sparse_head_and_loss_fn = False
        reward_model_config.fused_linear = model_args.fused_linear
        print(f"Loading Reward model with config:\n\t{reward_model_config}\n")

        if training_args.eval_mode is not None:
            config = copy.deepcopy(reward_model_config)
            if training_args.eval_mode == "single":
                config.tensor_parallel_degree = -1
                config.tensor_parallel_rank = 0
            if not training_args.autotuner_benchmark:
                reward_model = LlamaModelForScore.from_pretrained(
                    model_args.reward_model_name_or_path,
                    config=config,
                    score_type="reward",
                    do_normalize=False,
                )
            else:
                reward_model = LlamaModelForScore.from_config(
                    config,
                    score_type="reward",
                    do_normalize=False,
                )
        else:
            if not training_args.autotuner_benchmark:
                reward_model = model_class_score.from_pretrained(
                    model_args.reward_model_name_or_path,
                    config=reward_model_config,
                    score_type="reward",
                    do_normalize=False,
                )
            else:
                reward_model = model_class_score.from_config(
                    reward_model_config,
                    score_type="reward",
                    do_normalize=False,
                )

        logger.info(f"{runtime_timer.log()}")
        reward_tokenizer = AutoTokenizer.from_pretrained(
            model_args.reward_model_name_or_path,
            model_max_length=data_args.max_length,
            padding_side="right",
            tokenizer_alpha=model_args.reward_tokenizer_alpha,
        )
        llm_utils.init_chat_template(reward_tokenizer, model_args.reward_model_name_or_path, model_args.chat_template)
    else:
        reward_tokenizer = actor_tokenizer
        reward_model = model_args.reward_server
    if training_args.rl_algorithm == "ppo":
        # critic model
        runtime_timer.start("Reward critic model loading time")
        if model_args.reward_critic_model_name_or_path is None:
            model_args.reward_critic_model_name_or_path = model_args.reward_model_name_or_path
            reward_critic_model = model_class_score.from_config(
                reward_model_config,
                dtype=dtype,
                score_type="critic",
                do_normalize=False,
                clip_range_value=training_args.clip_range_value,
            )
            if not training_args.autotuner_benchmark:
                reward_critic_model.set_state_dict(reward_model.state_dict())
        else:
            if not training_args.autotuner_benchmark:
                reward_critic_model = model_class_score.from_pretrained(
                    model_args.reward_critic_model_name_or_path,
                    config=reward_model_config,
                    score_type="critic",
                    do_normalize=False,
                    clip_range_value=training_args.clip_range_value,
                )
            else:
                reward_critic_model = model_class_score.from_config(
                    reward_model_config,
                    score_type="critic",
                    do_normalize=False,
                    clip_range_value=training_args.clip_range_value,
                )
        logger.info(f"{runtime_timer.log()}")
        reward_critic_tokenizer = AutoTokenizer.from_pretrained(
            model_args.reward_critic_model_name_or_path,
            model_max_length=data_args.max_length,
            padding_side="left",
            tokenizer_alpha=model_args.reward_critic_tokenizer_alpha,
        )
        llm_utils.init_chat_template(
            reward_critic_tokenizer, model_args.reward_critic_model_name_or_path, model_args.chat_template
        )
        if training_args.eval_mode is not None:
            config = copy.deepcopy(reward_critic_model.config)
            if training_args.eval_mode == "single":
                config.tensor_parallel_degree = -1
                config.tensor_parallel_rank = 0
            runtime_timer.start("Reward critic eval model loading time")
            reward_critic_eval_model = LlamaModelForScore.from_config(config)
            logger.info(f"{runtime_timer.log()}")
            # reward_critic_eval_model =  AutoModelForScore.from_pretrained(
            #     model_args.reward_critic_model_name_or_path,config=model_config
            # )
            # cleanup_tensor_space(reward_critic_eval_model.state_dict())
        else:
            reward_critic_eval_model = None

    for tokenizer in [
        actor_tokenizer,
        reward_tokenizer,
        reward_critic_tokenizer if training_args.rl_algorithm == "ppo" else None,
    ]:
        if tokenizer and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_args.should_load_dataset:
        train_ds = PromptOnlyDataset(
            data_args.parsed_train_datasets, tokenizer=actor_tokenizer, use_rm_server=training_args.use_rm_server
        )
        if data_args.eval_datasets is None and data_args.eval_split_ratio:
            train_ds, dev_ds = train_ds.split_train_test(split_ratio=data_args.eval_split_ratio)
        elif data_args.eval_datasets is not None:
            dev_ds = PromptOnlyDataset(
                data_args.parsed_eval_datasets, tokenizer=actor_tokenizer, use_rm_server=training_args.use_rm_server
            )
        else:
            dev_ds = None

        ptx_ds = (
            SupervisedDataset(data_args.parsed_ptx_datasets, tokenizer=actor_tokenizer)
            if data_args.ptx_datasets is not None
            else None
        )
        if ptx_ds is not None:
            # PretrainingCriterion requires shifted inputs and labels
            ptx_ds.get_collator = types.MethodType(partial(ptx_ds.get_collator.__func__, shift=True), ptx_ds)

    if "freeze_model" in training_args.offload_level:
        offload_tensor_to_cpu((actor_reference_model, "freeze_model"))
        if training_args.rl_algorithm == "ppo":
            offload_tensor_to_cpu((reward_model, "freeze_model"))
        if actor_eval_model is not None:
            offload_tensor_to_cpu((actor_eval_model, "freeze_model"))
        if training_args.rl_algorithm == "ppo" and reward_critic_eval_model is not None:
            offload_tensor_to_cpu((reward_critic_eval_model, "freeze_model"))
        # NOTE(gongenlei): release memory_reserved_size to equal to memory_allocated_size
        paddle.device.cuda.empty_cache()

    def compute_metrics(eval_preds):
        accuracy = (eval_preds.predictions == 3).astype("float32").mean().item()
        return {
            "accuracy": accuracy,
        }

    trainer = PPOTrainer(
        #  (policy_model, reference_model, reward_model, value_model)
        #   policy_model, sft_model,       reward_model, value_model
        #  (policy_model, reference_model, reward_model, value_model,
        #  (policy_model, reference_model, reward_model, value_model, policy_eval_model, value_eval_model
        #  (actor_model, actor_reference_model, reward_model, reward_critic_model, actor_eval_model,
        #   reward_critic_eval_model
        model=(
            actor_model,
            actor_reference_model,
            reward_model,
            reward_critic_model if training_args.rl_algorithm == "ppo" else None,
            actor_eval_model,
            reward_critic_eval_model if training_args.rl_algorithm == "ppo" else None,
        ),
        args=training_args,
        train_dataset=(train_ds if training_args.do_train and training_args.should_load_dataset else None),
        eval_dataset=(dev_ds if training_args.do_eval and training_args.should_load_dataset else None),
        ptx_dataset=ptx_ds,
        tokenizer=(
            actor_tokenizer,
            actor_tokenizer,
            reward_tokenizer,
            reward_critic_tokenizer if training_args.rl_algorithm == "ppo" else None,
        ),
        data_collator=train_ds.get_collator(),
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
        from paddlenlp.trainer import EarlyStoppingCallback

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
            runtime_timer.start("Model saving time")
            trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
            if paddle.distributed.get_world_size() > 1:
                paddle.distributed.barrier()
            logger.info(f"{runtime_timer.log()}")
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
