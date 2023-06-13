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

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


# python bloom_nl2sql.py --model_name_or_path bigscience/bloomz-3b --train_file nl2sql/train.jsonl --validation_file nl2sql/dev.jsonl
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
    parser = PdArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        fp16_opt_level="O1",
        logging_steps=50,
        output_dir="outputs",
    )
    if model_args.lora:
        dtype = model.config.dtype
        if training_args.fp16_opt_level == "O2":
            if training_args.fp16:
                dtype = "float16"
            if training_args.bf16:
                dtype = "bfloat16"
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
        data_collator=DataCollatorForSeq2Seq(return_tensors="pd", tokenizer=tokenizer),
    )
    trainer.train()


if __name__ == "__main__":
    main()
