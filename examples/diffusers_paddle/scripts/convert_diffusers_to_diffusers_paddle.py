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

paddle.set_device("cpu")
from collections import OrderedDict
import copy
import torch
from diffusers_paddle import StableDiffusionPipeline as PaddleStableDiffusionPipeline, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
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
        # 过滤掉ignore
        if any(i in k for i in ignore):
            continue
        oldk = copy.deepcopy(k)
        # 批量替换名字
        is_transpose = False
        if v.ndim == 2:
            if "embeddings" in oldk or "norm" in oldk or 'concept_embeds' in oldk or 'special_care_embeds' in oldk:
                pass
            else:
                v = v.t()
                is_transpose = True
        for oldname, newname in old2new.items():
            k = k.replace(oldname, newname).replace(".vision_model.", ".")

        # pytorch的是0d的tensor，paddle的是1d tensor所以要reshape。这要注意。
        if k == "logit_scale": v = v.reshape((1, ))
        if "vision_model" in k: k = "clip." + k
        if "text_model" in k: k = "clip." + k
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

    scheduler_config = dict(pytorch_pipe.scheduler.config)
    class_name = scheduler_config.pop("_class_name")
    version = scheduler_config.pop("_diffusers_version")
    new_scheduler = eval(class_name)(**scheduler_config)

    paddle_pipe = PaddleStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", scheduler=new_scheduler)
    paddle_pipe.vae.set_dict(new_vae)
    paddle_pipe.unet.set_dict(new_unet)
    paddle_pipe.text_encoder.set_dict(new_text_encoder)
    paddle_pipe.safety_checker.set_dict(new_safety_checker)
    return paddle_pipe


if __name__ == "__main__":
    # model_name为Huggingface.co上diffusers权重地址。
    paddle_pipe = convert_model(model_name="CompVis/stable-diffusion-v1-4")
    paddle_pipe.save_pretrained("./stable-diffusion-v1-4-paddle")
