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

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

lora = True
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
if lora:
    target_modules = ["query_key_value"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=target_modules, r=8, lora_alpha=32, lora_dropout=0.1
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


dataset = load_dataset("json", data_files={"train": "nl2sql/train.jsonl", "dev": "nl2sql/dev.jsonl"})
dataset = dataset.map(lambda example: preprocess_function(example))

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    args=TrainingArguments(
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        output_dir="outputs",
    ),
    data_collator=DataCollatorForSeq2Seq(return_tensors="pt", tokenizer=tokenizer),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
