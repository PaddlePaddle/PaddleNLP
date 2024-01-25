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

import json
import os
import sys

import paddle
from data.collator import DPODataCollatorWithPadding
from trainer.dpo_trainer import DPOTrainer
from utils.argument import DataArgument, ModelArgument, TrainingArguments
from utils.utils import get_lora_target_modules

from paddlenlp.datasets import load_dataset
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)
from paddlenlp.utils.log import logger


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


def main():
    # Arguments
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    ###############
    # Step1: Load datasets
    # todo(lugimzzz): support mixer dataset
    ###############
    if os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")) and os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev.json")
    ):
        train_ds, dev_ds = load_dataset(
            "json",
            data_files={
                "train": os.path.join(data_args.dataset_name_or_path, "train.json"),
                "dev": os.path.join(data_args.dataset_name_or_path, "dev.json"),
            },
            lazy=data_args.lazy,
        )
    elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train")) and os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev")
    ):
        import glob

        train_files = glob.glob(os.path.join(data_args.dataset_name_or_path, "train", "*.json"))
        dev_files = glob.glob(os.path.join(data_args.dataset_name_or_path, "dev", "*.json"))
        train_ds, dev_ds = load_dataset(
            "json", data_files={"train": train_files, "dev": dev_files}, lazy=data_args.lazy
        )
    else:
        raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###############
    # Step2: Load models & tokenizer
    ###############
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    if training_args.pipeline_parallel_degree > 1:
        if data_args.eval_with_do_generation and training_args.do_eval:
            raise ValueError("Plese set eval_with_do_generation to false in pipeline parallel mode.")
        from models.modeling_pp import LlamaForCausalLMPipe

        model = LlamaForCausalLMPipe.from_pretrained(
            model_args.model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            use_flash_attention=model_args.use_flash_attention,
            dtype=dtype,
        )
        if model_args.lora:
            ref_model = None
        else:
            ref_model = LlamaForCausalLMPipe.from_pretrained(
                model_args.model_name_or_path,
                tensor_parallel_output=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                tensor_parallel_rank=training_args.tensor_parallel_rank,
                use_flash_attention=model_args.use_flash_attention,
                dtype=dtype,
            )
    else:
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            dtype=dtype,
        )
        if hasattr(model_config, "use_flash_attention"):
            model_config.use_flash_attention = model_args.use_flash_attention

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
        )
        if model_args.lora:
            ref_model = None
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=model_config,
            )

    if model_args.lora:
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_alpha,
                merge_weights=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # todo(lugimzzz): support chat template
    data_collator = DPODataCollatorWithPadding(
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        max_prompt_length=data_args.max_prompt_length,
    )

    #########################
    # Step3: Instantiate DPO trainer
    #########################
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        data_collator=data_collator,
    )
    ###############
    # Step4: Training & Evaluation
    ###############
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
