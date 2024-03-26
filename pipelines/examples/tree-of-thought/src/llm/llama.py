# coding=utf8, ErnestinaQiu

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
import time

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

llm_config = {
    "llama-2-7b": "meta-llama/Llama-2-7b",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat",
    "llama-2-13b": "meta-llama/Llama-2-13b",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat",
    "llama-2-70b": "meta-llama/Llama-2-70b",
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat",
    "llama-7b": "facebook/llama-7b",
    "llama-13b": "facebook/llama-13b",
    "llama-30b": "facebook/llama-30b",
    "llama-65b": "facebook/llama-65b",
    "ziqingyang/chinese-llama-7b": "ziqingyang/chinese-llama-7b",
    "ziqingyang/chinese-llama-13b": "ziqingyang/chinese-llama-13b",
    "ziqingyang/chinese-alpaca-7b": "ziqingyang/chinese-alpaca-7b",
    "ziqingyang/chinese-alpaca-13b": "ziqingyang/chinese-alpaca-13b",
    "idea-ccnl/ziya-llama-13b-v1": "idea-ccnl/ziya-llama-13b-v1",
    "linly-ai/chinese-llama-2-7b": "linly-ai/chinese-llama-2-7b",
    "linly-ai/chinese-llama-2-13b": "linly-ai/chinese-llama-2-13b",
    "baichuan-inc/Baichuan-7B": "baichuan-inc/Baichuan-7B",
    "baichuan-inc/Baichuan-13B-Base": "baichuan-inc/Baichuan-13B-Base",
    "baichuan-inc/Baichuan-13B-Chat": "baichuan-inc/Baichuan-13B-Chat",
    "baichuan-inc/Baichuan2-7B-Base": "baichuan-inc/Baichuan2-7B-Base",
    "baichuan-inc/Baichuan2-7B-Chat": "baichuan-inc/Baichuan2-7B-Chat",
    "baichuan-inc/Baichuan2-13B-Base": "baichuan-inc/Baichuan2-13B-Base",
    "baichuan-inc/Baichuan2-13B-Chat": "baichuan-inc/Baichuan2-13B-Chat",
    "FlagAlpha/Llama2-Chinese-7b-Chat": "FlagAlpha/Llama2-Chinese-7b-Chat",
    "FlagAlpha/Llama2-Chinese-13b-Chat": "FlagAlpha/Llama2-Chinese-13b-Chat",
}


class llamaChatCompletion:
    global llm_config

    def __init__(self, model="llama-2-7b-chat") -> None:
        config_path = llm_config[model]
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.generator = AutoModelForCausalLM.from_pretrained(config_path, dtype="float16")
        self.tokenizer.init_chat_template(
            os.path.join(os.getcwd(), "pipelines", "examples", "tree-of-thought", "src", "llm", "chat_template.json")
        )
        self.query = []
        self.query_count = 0

    def create(self, messages, temperature=0.6, top_p=0.9, max_gen_len=512):
        """
        Entry point of the program for generating text using a pretrained model.

        Args:
            messages (list): There are two roles including "system" and "user".
            --Example  [[{"role": "user", "content": "what is the recipe of mayonnaise?"}, {"role": "system", "content": "Always answer with Haiku"}]]
            ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
            tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
            temperature (float, optional): The temperature value for controlling randomness in generation.
                Defaults to 0.6.
            top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
                Defaults to 0.9.
            max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512. Max length is 4096
            max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
            max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
                set to the model's max sequence length. Defaults to None.
        """
        completion = {
            "choices": [],
            "created": time.time(),
            "id": "llama2_{}".format(int(time.time())),
            "model": self.model_name,
            "object": "chat.completion",
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }

        for i in range(len(messages)):
            one_mes = messages[i][0]
            assert len(messages[i]) == 1
            mes = one_mes["content"]
            self.query.append([mes])
            self.query_count += len(mes)
            while self.query_count > max_gen_len and len(self.query) > 2:
                pop_size = len("".join(self.query.pop(0)))
                self.query_count -= pop_size
            input_features = self.tokenizer.apply_chat_template(self.query, return_tensors="pd")
            outputs = self.generator.generate(
                **input_features,
                decode_strategy="greedy_search",
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_gen_len,
            )
            out_0 = self.tokenizer.batch_decode(outputs[0])
            self.query[-1].append(out_0[0])
            self.query_count += len(out_0[0])
            if i == len(messages) - 1:
                finish_reason = "stop"
            else:
                finish_reason = "length"
            tmp = {
                "finish_reason": finish_reason,
                "index": i,
                "message": {"content": "", "role": ""},
            }
            tmp["message"]["role"] = "llm"
            tmp["message"]["content"] = out_0
            completion["choices"].append(tmp)

        return completion
