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

import sys

import numpy as np

inputs = [
    np.load("torch_cache/inputs/input_ids.npy", allow_pickle=True),
    # np.load("torch_cache/inputs/input_ids_1.npy", allow_pickle=True),
]


def run_mp_paddle():
    import paddle

    from paddlenlp.transformers import ChatGLMForConditionalGeneration

    tensor_parallel_degree = paddle.distributed.get_world_size()
    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tensor_parallel_degree,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    paddle.distributed.fleet.init(is_collective=True, strategy=strategy)

    hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
    mp_group = hcg.get_model_parallel_group()
    tensor_parallel_rank = mp_group.rank

    paddle.set_default_dtype("float32")
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "torch_cache",  # "/root/paddlejob/workspace/GLM/ChatGLM-6B/",
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        dtype="float32",
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
    )
    model.eval()
    results = model(input_ids=paddle.to_tensor(inputs[0]), return_dict=True)
    print("results:", results)
    print("logits:", results.logits.abs().mean().item())


def run_paddle():
    import paddle

    from paddlenlp.transformers import ChatGLMForConditionalGeneration

    paddle.set_default_dtype("float32")
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "torch_cache",  # "/root/paddlejob/workspace/GLM/ChatGLM-6B/",
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        dtype="float32",
    )
    model.eval()
    results = model(input_ids=paddle.to_tensor(inputs[0]), return_dict=True)
    print("results:", results)
    print("logits:", results.logits.abs().mean().item())


def run_torch():
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

    results = model(input_ids=torch.tensor(inputs[0]).to(model.device), return_dict=True)
    print("results:", results)
    print("logits:", results.logits.abs().mean().item())


def run_generate():
    import paddle
    from modeling import ChatGLMForConditionalGeneration

    paddle.set_default_dtype("float16")
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "torch_cache", load_state_as_np=True, low_cpu_mem_usage=True, dtype="float16"
    )
    model.eval()

    from tokenizer import ChatGLMTokenizer

    tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b")
    inputs = tokenizer("你好", return_tensors="pd")
    print(inputs)
    outputs = model.generate(**inputs, max_length=2048)
    print(outputs)
    print(tokenizer.decode(outputs[0].tolist()))

    prompt = tokenizer.prepare_query_for_chat(
        query="晚上睡不着应该怎么办", history=[("你好", "你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。")]
    )
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pd")
    outputs = model.generate(**inputs, max_length=2048)
    print(outputs)
    print(tokenizer.decode(outputs[0].tolist()))


if __name__ == "__main__":
    if sys.argv[1] == "p":
        run_paddle()
    elif sys.argv[1] == "m":
        run_mp_paddle()
    elif sys.argv[1] == "t":
        run_torch()
    elif sys.argv[1] == "g":
        run_generate()
