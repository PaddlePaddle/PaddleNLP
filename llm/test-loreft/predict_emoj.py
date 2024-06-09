import sys
import json
import os
import sys
import paddle
import paddlenlp.reft.pareft as pareft
import paddlenlp.reft.pavenv as pavenv
from paddlenlp.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from paddlenlp.trainer import TrainingArguments


dtype = paddle.bfloat16
# 用户prompt
prompt_no_input_template = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

%s [/INST]
"""
device = "gpu"




model_name_or_path = "meta-llama/Llama-2-7b"

# get tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False
)
tokenizer.pad_token = tokenizer.unk_token



# load model
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype)


# load_directory = "/home/ldn/baidu/pyreft/paddle-version/loreft/tmp/intervenable_model"
load_directory = "/home/ldn/baidu/pyreft/paddle-version/apr/PaddleNLP/llm/test-loreft/tmp/intervenable_model"
reft_model = pareft.ReftModel.load(
    load_directory,
    model,
)


# 预测
# instruction = "Which dog breed do people think is cuter, poodle or doodle?"
instruction = "Who are you?"

# tokenize and prepare the input
prompt = prompt_no_input_template % instruction
prompt = tokenizer(prompt, return_tensors="pd")

base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
_, reft_response = reft_model.generate(
    prompt,
    unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True,
    max_new_tokens=512,
    # do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True,
)
# print(reft_response[0])
# print(tokenizer.decode(reft_response[0][0], skip_special_tokens=True))
print(tokenizer.batch_decode(reft_response[0], skip_special_tokens=True))

