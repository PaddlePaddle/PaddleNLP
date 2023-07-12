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

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.profiler as profiler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlamaTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModel,
)
import numpy as np
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
    english: Optional[bool] = field(default=False, metadata={"help": "whether to english benchmark dataset"})
    profiler: Optional[bool] = field(default=False, metadata={"help": "whether to use profiler"})


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    if "llama" in model_args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path,
                                                    use_fast=False)
        tokenizer.pad_token_id = 0
    elif "chatglm" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if 'chatglm' in model_args.model_name_or_path:
        # Add empty_init=False for zero3 training, refer to https://github.com/THUDM/ChatGLM-6B/issues/530
        model = AutoModel.from_pretrained(model_args.model_name_or_path, 
                                        empty_init=False if training_args.deepspeed is not None else True,
                                        trust_remote_code=True, torch_dtype="auto")
       
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype="auto")

    if model_args.lora:
        if "llama" in model_args.model_name_or_path:
            target_modules = ["q_proj", "k_proj", "v_proj"]
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
    dataset = dataset["train"].select(range(10000))
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
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-training-trainer'),
                profile_memory=True,
                with_stack=True,
            )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer, 
        train_dataset=dataset,
        callbacks=[ProfilerCallback(prof=prof)] if model_args.profiler else [],
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(return_tensors="pt", tokenizer=tokenizer),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    train_metrics = trainer.train()
    tokens_per_second = trainer.total_observed_tokens / train_metrics.metrics["train_runtime"]
    effective_tokens_per_second = total_effective_tokens / train_metrics.metrics["train_runtime"]
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
