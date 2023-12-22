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

from src.llm import llamaChatCompletion

completion_tokens = prompt_tokens = 0


def completions_with_backoff(**kwargs):
    chatter = kwargs["chatter"]
    print(kwargs["messages"])
    return chatter.create(messages=kwargs["messages"], temperature=kwargs["temperature"])


def gpt(prompt, model="llama-2-7b-chat", temperature=0.6, max_tokens=1000, n=1, stop=None) -> list:
    messages = [[{"role": "user", "content": prompt}]]
    return chatgpt(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=stop,
    )


def chatgpt(messages, model="llama-2-7b-chat", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    models_included = [
        "llama-2-7b-chat",
        "llama-2-7b",
        "llama2-13b-chat",
        "llama2-13b",
        "llama2-70b-chat",
        "llama2-70b",
    ]
    if model in models_included:
        chatter = llamaChatCompletion(model)
    else:
        print(f"Not support for llm {model}, and use llama-2-7b-chat instead.")
        chatter = llamaChatCompletion(model="llama-2-7b-chat")

    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        print("messages: {}".format(messages))
        res = completions_with_backoff(
            chatter=chatter,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
        )
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    outputs = outputs[0]
    return outputs


def gpt_usage(backend="llama-2-7b-chat"):
    global completion_tokens, prompt_tokens
    cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }
