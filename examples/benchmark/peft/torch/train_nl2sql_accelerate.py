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

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


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
    accelerator = Accelerator()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        fp16_opt_level="O2",
        logging_steps=50,
        output_dir="outputs",
    )

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

    data_collator = DataCollatorForSeq2Seq(return_tensors="pt", tokenizer=tokenizer)
    dataset = load_dataset("json", data_files={"train": data_args.train_file, "dev": data_args.validation_file})
    with accelerator.main_process_first():
        dataset = dataset.map(lambda example: preprocess_function(example), remove_columns=["src", "tgt"])
    accelerator.wait_for_everyone()

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=lambda x: data_collator(x),
        batch_size=training_args.per_device_train_batch_size,
        pin_memory=True,
    )
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * training_args.num_train_epochs),
    )

    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )

    for epoch in range(training_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    main()
