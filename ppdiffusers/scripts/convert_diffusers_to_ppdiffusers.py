# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import argparse

paddle.set_device("cpu")
from collections import OrderedDict
import copy
import torch
from ppdiffusers import StableDiffusionPipeline as PaddleStableDiffusionPipeline
# pip install diffusers
from diffusers import StableDiffusionPipeline as PytorchStableDiffusionPipeline


def convert_vae_to_paddlenlp(vae, dtype="float32"):
    need_transpose = []
    for k, v in vae.named_modules():
        if isinstance(v, torch.nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae = OrderedDict()
    for k, v in vae.state_dict().items():
        if k not in need_transpose:
            new_vae[k] = v.numpy().astype(dtype)
        else:
            new_vae[k] = v.t().numpy().astype(dtype)
    return new_vae


def convert_unet_to_paddlenlp(unet, dtype="float32"):
    need_transpose = []
    for k, v in unet.named_modules():
        if isinstance(v, torch.nn.Linear):
            need_transpose.append(k + ".weight")
    new_unet = OrderedDict()
    for k, v in unet.state_dict().items():
        if k not in need_transpose:
            new_unet[k] = v.numpy().astype(dtype)
        else:
            new_unet[k] = v.t().numpy().astype(dtype)
    return new_unet


def convert_hf_clip_to_paddlenlp(clip, dtype="float32"):
    new_model_state = OrderedDict()
    old2new = {
        ".encoder.": ".transformer.",
        ".layer_norm": ".norm",
        ".mlp.": ".",
        ".fc1.": ".linear1.",
        ".fc2.": ".linear2.",
        ".final_layer_norm.": ".ln_final.",
        ".embeddings.": ".",
        ".position_embedding.": ".positional_embedding.",
        ".patch_embedding.": ".conv1.",
        "visual_projection.weight": "vision_projection",
        "text_projection.weight": "text_projection",
        ".pre_layrnorm.": ".ln_pre.",
        ".post_layernorm.": ".ln_post."
    }
    ignore = ["position_ids"]

    for k, v in clip.state_dict().items():
        if any(i in k for i in ignore):
            continue
        oldk = copy.deepcopy(k)
        is_transpose = False
        if v.ndim == 2:
            if "embeddings" in oldk or "norm" in oldk or 'concept_embeds' in oldk or 'special_care_embeds' in oldk:
                pass
            else:
                v = v.t()
                is_transpose = True
        for oldname, newname in old2new.items():
            k = k.replace(oldname, newname).replace(".vision_model.", ".")

        if k == "logit_scale": v = v.reshape((1, ))
        if "vision_model" in k: k = "clip." + k
        new_model_state[k] = v.numpy().astype(dtype)
        print(f"Convert {oldk} -> {k} | {v.shape}, is_transpose {is_transpose}")
    return new_model_state


def convert_model(model_name):
    pytorch_pipe = PytorchStableDiffusionPipeline.from_pretrained(
        model_name, use_auth_token=True)
    new_vae = convert_vae_to_paddlenlp(pytorch_pipe.vae)
    new_unet = convert_unet_to_paddlenlp(pytorch_pipe.unet)
    new_text_encoder = convert_hf_clip_to_paddlenlp(pytorch_pipe.text_encoder)
    new_safety_checker = convert_hf_clip_to_paddlenlp(
        pytorch_pipe.safety_checker)

    paddle_pipe = PaddleStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4")
    paddle_pipe.vae.set_dict(new_vae)
    paddle_pipe.unet.set_dict(new_unet)
    paddle_pipe.text_encoder.set_dict(new_text_encoder)
    paddle_pipe.safety_checker.set_dict(new_safety_checker)
    return paddle_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="stable-diffusion-v1-5-ppdiffusers",
        help="The model output path.",
    )
    args = parser.parse_args()
    paddle_pipe = convert_model(args.pretrained_model_name_or_path)
    paddle_pipe.save_pretrained(args.output_path)
