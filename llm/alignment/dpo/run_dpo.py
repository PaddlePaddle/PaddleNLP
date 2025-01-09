# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

""" Training DPO """

import os
import sys
import time
from functools import partial

import paddle
from dpo_argument import (
    DPOConfig,
    DPODataArgument,
    DPOModelArgument,
    DPOTrainingArguments,
)

from paddlenlp.datasets import (
    ZeroPaddingIterableDataset,
    ZeroPaddingMapDataset,
    load_dataset,
)
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaForCausalLMPipe,
    Qwen2ForCausalLM,
    Qwen2ForCausalLMPipe,
)
from paddlenlp.transformers.configuration_utils import LlmMetaConfig
from paddlenlp.trl import (
    DPOTrainer,
    calculate_effective_tokens,
    preference_collate_fn,
    preprocess_preference_data,
)
from paddlenlp.trl.llm_utils import get_lora_target_modules
from paddlenlp.utils.log import logger

flash_mask_support_list = [Qwen2ForCausalLM, Qwen2ForCausalLMPipe, LlamaForCausalLM, LlamaForCausalLMPipe]


def main():
    """main"""
    parser = PdArgumentParser((DPOModelArgument, DPODataArgument, DPOTrainingArguments, DPOConfig))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, dpo_config = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args, dpo_config = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    set_seed(training_args.seed)
    if dpo_config.loss_type == "orpo":
        dpo_config.reference_free = True
        dpo_config.sft_loss_ratio = 1.0
        dpo_config.loss_type = "or"
        logger.info("orpo loss_type is equal to sft_loss + pref_loss_ratio * or_loss.")
    if dpo_config.loss_type in ["or", "simpo"] and not dpo_config.reference_free:
        dpo_config.reference_free = True
        logger.warning(f"{dpo_config.loss_type} loss_type only supports reference_free. Set reference_free to True.")
    if training_args.pipeline_parallel_degree > 1:
        assert (
            hasattr(training_args, "pipeline_parallel_config")
            and "enable_clear_every_step_cache" in training_args.pipeline_parallel_config
        ), "Should set '--pipeline_parallel_config enable_clear_every_step_cache' in bash script for pp."
    if training_args.sequence_parallel:
        if training_args.pipeline_parallel_degree > 1:
            assert (
                hasattr(training_args, "pipeline_parallel_config")
                and "disable_partial_send_recv" in training_args.pipeline_parallel_config
            ), "Should set '--pipeline_parallel_config disable_partial_send_recv' in bash script for pp with sp."
        if training_args.tensor_parallel_degree <= 1:
            training_args.sequence_parallel = False
            logger.info("Tensor_parallel_degree = 1. Set sequence_parallel to False.")
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(dpo_config, "DPOConfig")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: "
        f"{training_args.world_size}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load model & tokenizer.")

    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, dtype=dtype)
    LlmMetaConfig.set_llm_config(model_config, training_args)

    if not dpo_config.reference_free and not dpo_config.lora:
        ref_model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, dtype=dtype)
        LlmMetaConfig.set_llm_config(ref_model_config, training_args)

    if training_args.pipeline_parallel_degree > 1:
        model_class = AutoModelForCausalLMPipe
        model_config.dpo_config = dpo_config
    else:
        model_class = AutoModelForCausalLM
    if not training_args.autotuner_benchmark or model_args.weight_quantize_algo is not None:
        model = model_class.from_pretrained(model_args.model_name_or_path, config=model_config)
        # for DPO save
        if not dpo_config.reference_free and not dpo_config.lora:
            ref_model = model_class.from_config(ref_model_config)
            ref_model.set_state_dict(model.state_dict())
        else:
            ref_model = None
    else:
        model = model_class.from_config(model_config)
        if not dpo_config.reference_free and not dpo_config.lora:
            ref_model = model_class.from_config(ref_model_config)
        else:
            ref_model = None
    if training_args.pipeline_parallel_degree > 1:
        model.config.dpo_config = None
    if model_args.flash_mask and not model.config.use_flash_attention:
        logger.warning("`flash_mask` must use with zero padding and flash attention.")
        model.config.use_flash_attention = True

    if model_args.flash_mask and not any(isinstance(model, cls) for cls in flash_mask_support_list):
        raise NotImplementedError(f"{model.__class__} not support flash mask.")

    if model_args.tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # TODO: support chat template in next pr
    tokenizer.chat_template = None
    logger.info("Loading model & tokenizer successfully !")

    if dpo_config.lora:
        if training_args.sharding_parallel_degree > 1:
            assert (
                "enable_stage1_overlap" not in training_args.sharding_parallel_config
            ), "Currently not support enabling sharding_stage1_overlap in lora mode."
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            if model_args.rslora_plus:
                model_args.rslora = True
                model_args.lora_plus_scale = 4
                model_args.lora_alpha = 4
            if model_args.weight_quantize_algo is not None:
                if model_args.rslora or model_args.lora_plus_scale != 1.0:
                    logger.info("Weight quantization is not supported in LoRA+ and RsLoRA.")
            if model_args.lora_alpha == -1:
                if model_args.rslora:
                    model_args.lora_alpha = 4
                else:
                    model_args.lora_alpha = 2 * model_args.lora_rank
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank if not model_args.rslora else 4,
                rslora=model_args.rslora,
                lora_plus_scale=model_args.lora_plus_scale,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
                use_quick_lora=model_args.use_quick_lora,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)

        model.print_trainable_parameters()

    logger.info("Start to create dataset")
    trans_func = partial(preprocess_preference_data, tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    if data_args.lazy:
        zero_padding_dataset = ZeroPaddingIterableDataset
    else:
        zero_padding_dataset = ZeroPaddingMapDataset
    if training_args.do_train and training_args.should_load_dataset:
        train_ds = load_dataset(
            "json",
            data_files=data_args.train_dataset_path,
            lazy=data_args.lazy,
        )[0]
        logger.info("Creating train Zero Padding Data Stream. This may take a few minutes.")
        train_ds = (
            zero_padding_dataset(
                train_ds.map(trans_func),
                tokenizer=tokenizer,
                max_length=data_args.max_seq_len,
                greedy_zero_padding=data_args.greedy_zero_padding,
            )
            if train_ds is not None
            else None
        )
    else:
        train_ds = None

    if training_args.do_eval and training_args.should_load_dataset:
        eval_ds = load_dataset(
            "json",
            data_files=data_args.dev_dataset_path,
            lazy=data_args.lazy,
        )[0]
        logger.info("Creating dev Zero Padding Data Stream. This may take a few minutes.")
        eval_ds = (
            zero_padding_dataset(
                eval_ds.map(trans_func),
                tokenizer=tokenizer,
                max_length=data_args.max_seq_len,
            )
            if eval_ds is not None
            else None
        )
    else:
        eval_ds = None
    logger.info("Creating dataset successfully ...")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        dpo_config=dpo_config,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=partial(
            preference_collate_fn,
            max_seq_len=data_args.max_seq_len,
        ),
        ignore_eos_token=True,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        if not training_args.autotuner_benchmark and not training_args.benchmark:
            trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
        if training_args.benchmark:
            total_effective_tokens, total_tokens = calculate_effective_tokens(
                training_args, train_ds, data_args.max_seq_len
            )
            effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
            total_tokens_per_second = total_tokens / train_result.metrics["train_runtime"]
            effective_ratio = 100 * total_effective_tokens / total_tokens
            logger.info(
                "[timelog] {}: {:.2f} % ({}) ".format(
                    "Effective ratio", effective_ratio, time.strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            logger.info(
                "[timelog] {}: {:.2f} token/s ({}) ".format(
                    "Effective tokens per second", effective_tokens_per_second, time.strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            logger.info(
                "[timelog] {}: {:.2f} token/s ({}) ".format(
                    "Tokens per second", total_tokens_per_second, time.strftime("%Y-%m-%d %H:%M:%S")
                )
            )

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
