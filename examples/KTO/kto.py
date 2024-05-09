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

from dataclasses import dataclass

from datasets import load_dataset
from kto_config import KTOConfig
from kto_trainer import KTOTrainer
from model_config import ModelConfig

from paddlenlp.peft import LoRAConfig
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    dataset_name: str = "trl-lib/kto-mix-14k"


if __name__ == "__main__":
    parser = PdArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    dtype = None
    if kto_args.fp16_opt_level == "O2":
        if kto_args.fp16:
            dtype = "float16"
        if kto_args.bf16:
            dtype = "bfloat16"
    else:
        dtype = "float32"

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained("llama2-7b-chat")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TODO(wugaosheng) adapt to chattemplte
    # If we are aligning a base model, we use ChatML as the default template
    # if tokenizer.chat_template is None:
    #     model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)

    # Apply chat template
    def format_dataset(example):
        # Add chattemple to prompt input
        chat_msg = [item["content"] for item in example["prompt"]]
        res = ""
        for i in range(0, len(chat_msg), 2):
            if i + 1 < len(chat_msg):
                res += f"<s>[INST] {chat_msg[i].strip()} [/INST] {chat_msg[i+1].strip()} </s>"
        pd_output = tokenizer.apply_chat_template(chat_msg[-1], tokenize=False)
        pd_output = res + pd_output
        pd_output = res + f"<s>[INST] {chat_msg[-1].strip()} [/INST]"
        example["prompt"] = pd_output

        # Add chattemple to completion
        chat_msg = ["Hi"] + [item["content"] for item in example["completion"]]
        res = ""
        for i in range(0, len(chat_msg), 2):
            if i + 1 < len(chat_msg):
                res += f"<s>[INST] {chat_msg[i].strip()} [/INST] {chat_msg[i+1].strip()} </s>"
        pd_output = res
        # remove fake user content
        example["completion"] = pd_output.split("[/INST]")[-1]
        return example

    formatted_dataset = dataset.map(format_dataset)

    if model_args.use_peft:
        target_modules = [
            ".*q_proj.*",
            ".*v_proj.*",
            ".*k_proj.*",
            ".*gate_proj.*",
            ".*up_proj.*",
            ".*o_proj.*",
            ".*down_proj.*",
        ]

        peft_config = LoRAConfig(target_modules=target_modules, r=model_args.lora_r, lora_alpha=model_args.lora_alpha)
    else:
        peft_config = None
    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train and push the model to the Hub
    kto_trainer.train()
