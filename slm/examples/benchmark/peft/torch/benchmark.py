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
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.profiler as profiler
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    LlamaTokenizer,
    TrainingArguments,
)
from utils import CustomTrainer, ProfilerCallback

"""
单卡
python benchmark.py --model_name_or_path bigscience/bloomz-7b1-mt  \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --evaluation_strategy no --save_strategy no \
    --fp16 --lora \
    --logging_steps 50 --output_dir outputs

多卡 deepspeed zero3
python -m torch.distributed.run --nproc_per_node=4 benchmark.py --deepspeed ds_config.json \
    --model_name_or_path bigscience/bloomz-7b1-mt  \
    --num_train_epochs 1 --per_device_train_batch_size 2 \
    --evaluation_strategy no --save_strategy no \
    --fp16 \
    --logging_steps 50 --output_dir outputs
"""


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(default=None, metadata={"help": "model name or local path"})
    lora: Optional[bool] = field(default=False, metadata={"help": "whether to use LoRA"})
    qlora: Optional[bool] = field(default=False, metadata={"help": "whether to use qLoRA"})
    english: Optional[bool] = field(default=False, metadata={"help": "whether to english benchmark dataset"})
    profiler: Optional[bool] = field(default=False, metadata={"help": "whether to use profiler"})
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})
    train_data_size: int = field(default=1000, metadata={"help": "Number of dataset for training"})


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    if "llama" in model_args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        tokenizer.pad_token_id = 0
    elif model_args.model_name_or_path in ["cerebras/Cerebras-GPT-13B", "stanford-crfm/levanter-gpt2-7B"]:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        tokenizer.pad_token_id = 0
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    if "chatglm" in model_args.model_name_or_path:
        # Add empty_init=False for zero3 training, refer to https://github.com/THUDM/ChatGLM-6B/issues/530
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            empty_init=False if training_args.deepspeed is not None else True,
            trust_remote_code=True,
            torch_dtype="auto",
        )

    else:
        if model_args.qlora:
            n_gpus = torch.cuda.device_count()
            max_memory = f"{model_args.max_memory_MB}MB"
            max_memory = {i: max_memory for i in range(n_gpus)}
            device_map = "auto"

            # if we are in a distributed setting, we need to set the device map and max memory per device
            if os.environ.get("LOCAL_RANK") is not None:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device_map = {"": local_rank}
                max_memory = {"": max_memory[local_rank]}

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype="auto",
                load_in_4bit=model_args.bits == 4,
                load_in_8bit=model_args.bits == 8,
                device_map=device_map,
                max_memory=max_memory,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=model_args.bits == 4,
                    load_in_8bit=model_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=model_args.double_quant,
                    bnb_4bit_quant_type=model_args.quant_type,
                ),
            )
        elif model_args.model_name_or_path in ["cerebras/Cerebras-GPT-13B", "stanford-crfm/levanter-gpt2-7B"]:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype="auto",
            )
    if model_args.lora:
        if "llama" in model_args.model_name_or_path:
            target_modules = ["q_proj", "k_proj", "v_proj"]
        elif model_args.model_name_or_path in ["cerebras/Cerebras-GPT-13B", "stanford-crfm/levanter-gpt2-7B"]:
            target_modules = [
                ".*c_attn.*",
                ".*q_attn.*",
                ".*c_proj.*",
                ".*c_fc.*",
            ]
        else:
            target_modules = ["query_key_value"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, target_modules=target_modules, r=8, lora_alpha=32, lora_dropout=0.0
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if model_args.lora and training_args.gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    def preprocess_function(example, max_src_length=256, max_tgt_length=384):
        inputs = example["instruction"]
        if "input" in example:
            inputs += example["input"]
        targets = example["output"]
        model_inputs = tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=False)

        labels = tokenizer(targets, max_length=max_tgt_length, truncation=True, return_attention_mask=False)
        labels_input_ids = labels["input_ids"] + [tokenizer.eos_token_id]

        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids
        return model_inputs

    if model_args.english:
        dataset = load_dataset("tatsu-lab/alpaca")
    else:
        dataset = load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")

    # select first 10k examples for benchmarking
    dataset = dataset["train"].select(range(model_args.train_data_size))
    dataset = dataset.map(
        lambda example: preprocess_function(example), remove_columns=["instruction", "input", "output"]
    )
    total_effective_tokens = sum([len(i["input_ids"]) for i in dataset]) * training_args.num_train_epochs

    if model_args.profiler:
        prof = profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("hf-training-trainer"),
            profile_memory=True,
            with_stack=True,
        )

    data_collator = DataCollatorForSeq2Seq(return_tensors="pt", tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        callbacks=[ProfilerCallback(prof=prof)] if model_args.profiler else [],
        args=training_args,
        data_collator=data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    train_metrics = trainer.train()
    tokens_per_second = trainer.total_observed_tokens / train_metrics.metrics["train_runtime"]
    effective_tokens_per_second = total_effective_tokens / train_metrics.metrics["train_runtime"]
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
