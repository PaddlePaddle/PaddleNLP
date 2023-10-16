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
from utils.argument import (
    DataArgument,
    GenerateArgument,
    ModelArgument,
    QuantArgument,
    TrainingArguments,
)
from utils.utils import get_lora_target_modules, get_prefix_tuning_params

from paddlenlp.datasets import load_dataset
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint
from paddlenlp.trainer.trainer_callback import TrainerState
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
    parser = PdArgumentParser((GenerateArgument, QuantArgument, ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(quant_args, "Quant")
    training_args.print_config(gen_args, "Generation")

    if sum([quant_args.do_ptq, quant_args.do_qat, quant_args.do_gptq, training_args.do_train]) > 1:
        raise ValueError(
            "--do_train, --do_ptq, --do_gptq and --do_qat cannot work at the same time. Please choose only one at a time"
        )

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
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

        ref_model = LlamaForCausalLMPipe.from_pretrained(
            model_args.ref_model_name_or_path,
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

        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.ref_model_name_or_path,
            config=model_config,
        )

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load the Stack-exchange paired dataset
    if data_args.dataset_name_or_path is None:
        raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")

    elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")) and os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev.json")
    ):
        # train_ds, dev_ds = load_dataset(
        #     "json",
        #     data_files={
        #         "train": os.path.join(data_args.dataset_name_or_path, "train.json"),
        #         "dev": os.path.join(data_args.dataset_name_or_path, "dev.json"),
        #     },
        #     lazy=data_args.lazy,
        # )
        train_ds = load_dataset(
            read_local_dataset,
            path=os.path.join(data_args.dataset_name_or_path, "train.json"),
            lazy=data_args.lazy,
        )
        dev_ds = load_dataset(
            read_local_dataset,
            path=os.path.join(data_args.dataset_name_or_path, "dev.json"),
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
        if data_args.task_name is not None:
            train_ds, dev_ds = load_dataset(
                data_args.dataset_name_or_path, data_args.task_name, splits=["train", "dev"]
            )
        else:
            train_ds, dev_ds = load_dataset(data_args.dataset_name_or_path, splits=["train", "dev"])

    if training_args.resume_from_checkpoint is not None and data_args.lazy:
        logger.info(
            f"Loading from '{training_args.resume_from_checkpoint}' with `lazy=True`, manually skipping dataset and setting `ignore_data_skip` to True."
        )
        training_args.ignore_data_skip = True
        state = TrainerState.load_from_json(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"))
        if state.trial_params is not None and "intokens_global_step" in state.trial_params:
            consumed_samples = state.trial_params["intokens_global_step"]
        else:
            consumed_samples = (
                state.global_step
                * training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.dataset_world_size
            )
        logger.info(
            f"Skipping the first {consumed_samples} samples to warmup the dataset from checkpoint '{training_args.resume_from_checkpoint}'."
        )
        train_ds = train_ds.skip(consumed_samples)

    # train_ds.data = train_ds.data[:1]
    # train_ds.new_data = train_ds.new_data[:1]
    dev_ds.data = dev_ds.data[:1000]
    dev_ds.new_data = dev_ds.new_data[:1000]

    if model_args.prefix_tuning:
        prefix_tuning_params = get_prefix_tuning_params(model)
        prefix_config = PrefixConfig(
            num_prefix_tokens=model_args.num_prefix_tokens,
            num_attention_heads=prefix_tuning_params["num_attention_heads"],
            num_hidden_layers=prefix_tuning_params["num_hidden_layers"],
            hidden_size=prefix_tuning_params["hidden_size"],
            multi_query_group_num=prefix_tuning_params["multi_query_group_num"],
            dtype=dtype,
        )
        model = PrefixModelForCausalLM(
            model=model,
            prefix_config=prefix_config,
            postprocess_past_key_value=prefix_tuning_params["postprocess_past_key_value"],
        )
        model.mark_only_prefix_as_trainable()
        model.print_trainable_parameters()

    if model_args.lora:
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank,
                merge_weights=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    # max_length = data_args.max_length if training_args.pipeline_parallel_degree > 1 else None
    # padding = "max_length" if training_args.pipeline_parallel_degree > 1 else True

    data_collator = DPODataCollatorWithPadding(
        tokenizer,
        max_length=data_args.max_length,
        max_prompt_length=data_args.max_prompt_length,
        padding_value=tokenizer.pad_token_id,
    )

    def compute_dpo_metrics(eval_pred):
        logits, metrics = eval_pred
        logs = {}
        keys = ["rewards_chosen", "rewards_rejected", "accuracy", "rewards_margins", "logps_rejected", "logps_chosen"]
        for i, met in enumerate(metrics):
            logs[keys[i]] = met.mean().item()
        return logs

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_ds,
        compute_metrics=compute_dpo_metrics,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        data_collator=data_collator,
    )
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if training_args.benchmark:
            total_effective_tokens = (
                sum([len(i["input_ids"]) for i in trainer.train_dataset]) * training_args.num_train_epochs
            )
            effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
            logger.info(f"Effective_Tokens_per_second: {effective_tokens_per_second} ")
            logger.info("Benchmark done.")
        else:
            trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
