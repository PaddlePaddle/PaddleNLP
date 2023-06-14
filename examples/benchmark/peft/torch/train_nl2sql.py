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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

"""
单卡
python train_nl2sql.py --model_name_or_path bigscience/bloomz-7b1-mt  \
    --train_file nl2sql/dev.jsonl --validation_file nl2sql/dev.jsonl \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --evaluation_strategy epoch --save_strategy epoch \
    --fp16 \
    --logging_steps 50 --output_dir outputs

多卡 deepspeed zero3
python -m torch.distributed.run --nproc_per_node=4 train_nl2sql.py --deepspeed ds_config.json \
    --model_name_or_path bigscience/bloomz-7b1-mt  \
    --train_file nl2sql/dev.jsonl --validation_file nl2sql/dev.jsonl \
    --num_train_epochs 1 --per_device_train_batch_size 2 \
    --evaluation_strategy epoch --save_strategy epoch \
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: str = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: str = field(default=None, metadata={"help": "The input evaluation data file (a text file).e)."})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    if model_args.lora:
        target_modules = ["query_key_value"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, target_modules=target_modules, r=8, lora_alpha=32, lora_dropout=0.0
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    def preprocess_function(example, max_src_length=512, max_tgt_length=256):
        inputs = example["src"][0]
        targets = example["tgt"][0]
        model_inputs = tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=False)
        labels = tokenizer(targets, max_length=max_tgt_length, truncation=True, return_attention_mask=False)
        labels_input_ids = labels["input_ids"] + [tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids

        return model_inputs

    dataset = load_dataset("json", data_files={"train": data_args.train_file, "dev": data_args.validation_file})
    dataset = dataset.map(lambda example: preprocess_function(example))

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(return_tensors="pt", tokenizer=tokenizer),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


if __name__ == "__main__":
    main()
