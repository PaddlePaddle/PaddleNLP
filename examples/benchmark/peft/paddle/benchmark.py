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
from utils import CustomTrainer

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

"""
单卡
python benchmark.py --model_name_or_path bigscience/bloomz-7b1-mt  \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --evaluation_strategy no --save_strategy no \
    --fp16 --fp16_opt_level O2 --lora \
    --logging_steps 50 --output_dir outputs

多卡mp
python -m paddle.distributed.launch --gpus "0,1,2,3" benchmark.py --model_name_or_path bigscience/bloomz-7b1-mt  \
    --num_train_epochs 1 --per_device_train_batch_size 8 \
    --evaluation_strategy no --save_strategy no \
    --fp16 --fp16_opt_level O2 --tensor_parallel_degree 4 \
    --logging_steps 50 --output_dir outputs

多卡sharding 3
python -m paddle.distributed.launch --gpus "0,1,2,3" benchmark.py --model_name_or_path bigscience/bloomz-7b1-mt  \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --evaluation_strategy no --save_strategy no \
    --fp16 --fp16_opt_level O2 \
    --sharding "stage3" --sharding_parallel_degree 4 \
    --logging_steps 50 --output_dir outputs
"""


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(default=None, metadata={"help": "model name or local path"})
    lora: Optional[bool] = field(default=False, metadata={"help": "whether to use LoRA"})


def main():
    parser = PdArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Set the dtype for loading model
    dtype = None
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if "llama" in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        # use_flash_attention=True,
        dtype=dtype,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        use_recompute=training_args.recompute,
    )

    if model_args.lora:
        # hardcode parameters for now
        lora_config = LoRAConfig(
            target_modules=[".*query_key_value.*"],
            r=8,
            lora_alpha=32,
            dtype=dtype,
        )
        model = LoRAModel(model, lora_config)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    def preprocess_function(example, max_src_length=512, max_tgt_length=512):
        inputs = example["instruction"]
        targets = example["output"]
        model_inputs = tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=False)
        labels = tokenizer(targets, max_length=max_tgt_length, truncation=True, return_attention_mask=False)
        labels_input_ids = labels["input_ids"] + [tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids

        return model_inputs

    dataset = load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")
    # select first 10k examples for benchmarking
    dataset = dataset["train"].select(range(10000))
    dataset = dataset.map(
        lambda example: preprocess_function(example), remove_columns=["instruction", "input", "output"]
    )
    total_effective_tokens = sum([len(i["input_ids"]) for i in dataset]) * training_args.num_train_epochs

    trainer = CustomTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(return_tensors="pd", tokenizer=tokenizer),
    )
    train_metrics = trainer.train()
    tokens_per_second = trainer.total_observed_tokens / train_metrics.metrics["train_runtime"]
    effective_tokens_per_second = total_effective_tokens / train_metrics.metrics["train_runtime"]
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
