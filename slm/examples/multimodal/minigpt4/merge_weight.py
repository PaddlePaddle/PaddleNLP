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

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"

import paddle
import torch

from paddlenlp.transformers import LlamaForCausalLM


def merge(args):
    model_dict = {}
    # load the first item: blip2-flan-t5-xxl
    state_dict = paddle.load(args.blip2_path)
    for n, p in state_dict.items():
        if n.startswith("vision_model") or n.startswith("qformer") or n == "query_tokens":
            model_dict[n] = p
    print("[1/3] load ViT, qformer and query_tokens from blip2-flan-t5-xxl done!")

    # load the second item: vicuna
    llama_model = LlamaForCausalLM.from_pretrained(args.vicuna_path)

    for n, p in llama_model.named_parameters():
        new_name = "language_model." + n
        model_dict[new_name] = p
    print("[2/3] load vicuna(llama typel) done!")

    # load the third item: minigpt4
    minigpt4_state_dict = torch.load(args.minigpt4_path)
    for n, p in minigpt4_state_dict["model"].items():
        if n.startswith("llama_model.model"):
            new_name = n.replace("llama_model.model", "language_model.llama")
            new_p = paddle.to_tensor(p.cpu().numpy())
            model_dict[new_name] = new_p

        if n.startswith("llama_proj"):
            new_name = n.replace("llama_proj", "language_projection")
            if n.endswith("weight"):
                new_p = paddle.to_tensor(p.cpu().numpy()).transpose([1, 0])
            else:
                new_p = paddle.to_tensor(p.cpu().numpy())
            model_dict[new_name] = new_p

    print("[3/3] load language_projection, some llama weights from minigpt4 done!")

    save_path = os.path.join(args.save_path, "model_state.pdparams")
    paddle.save(model_dict, save_path)
    print("The checkpoint of minigpt4 has been saved to :{}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--blip2_path", default="/blip2/dirname", type=str, help="The dir name of blip2-flan-t5-xxl.")
    parser.add_argument("--vicuna_path", default="/vicuna/dirname", type=str, help="The dir name of vicuna.")
    parser.add_argument(
        "--minigpt4_path", default="/minigpt4/prerained_minigpt4.pth", type=str, help="The checkpoint path of vicuna."
    )
    parser.add_argument("--save_path", default="/save/to/dirname", type=str, help="The saving path of minigpt4.")
    args = parser.parse_args()

    args.blip2_path = os.path.join(args.blip2_path, "model_state.pdparams")
    if not os.path.exists(args.blip2_path):
        raise ValueError("Not found the file: {}".format(args.blip2_path))
    if not os.path.isdir(args.vicuna_path):
        raise ValueError("It is not a directory: {}".format(args.vicuna_path))
    if not os.path.exists(args.minigpt4_path):
        raise ValueError("Not found the file: {}".format(args.minigpt4_path))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    merge(args)
