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

import argparse
import itertools
import os

import torch
from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training a text to image model script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="int8_images",
        help="The output directory for images.",
    )
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible running.")
    args = parser.parse_args()
    return args


args = parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
torch.manual_seed(args.seed)
pipe = StableDiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
)
pipe.safety_checker = None

prompt = "a photo of an astronaut riding a horse on mars"
# prompt = "illustration of close-up street view of gothic town, night, by peter mohrbacher, by alex andreev, by jacek yerka, large depth of field, super detailed, digital art, trending on artstation, minimalism"
# prompt = "a highly detailed portrait of a man with dark green hair and green glowing eyes, high detail clothing, concept art, anime, artstation, professional"
# prompt = "beautiful, young woman, cybernetic, cyberpunk, detailed gorgeous face, flowing hair, vaporwave aesthetic, synthwave , digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha"


def get_all_sub_module_names(module, prefix, handler=lambda x: True):
    r"""Returns a list of all submodule names in the module, including itself.

    Args:
        module: input module
        prefix: prefix of the current module, used as key in qconfig_dict

    Return:
        list of all submodule names in the module, including itself
    """
    ret = [prefix] if handler((prefix, module)) else []
    for name, child in module.named_children():
        module_prefix = prefix + "." + name if prefix else name
        ret.extend(get_all_sub_module_names(child, module_prefix, handler))
    return ret


linear_moduels = get_all_sub_module_names(pipe.unet, "", lambda x: isinstance(x[1], torch.nn.Linear))

mnames = []
indice = itertools.product(range(0, 4), range(3), range(1, 3))
for i, j, k in indice:
    mnames += [
        #         f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.attn{k}.to_q",
        #         f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.attn{k}.to_k",
        #         f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.attn{k}.to_v",
        #         f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.attn{k}.to_out.0",
        #         f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.ff.net.0.proj",
        f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.ff.net.2",
    ]
    mnames += [
        #         f"down_blocks.{i}.attentions.{j}.transformer_blocks.0.attn{k}.to_q",
        #         f"down_blocks.{i}.attentions.{j}.transformer_blocks.0.attn{k}.to_k",
        #         f"down_blocks.{i}.attentions.{j}.transformer_blocks.0.attn{k}.to_v",
        #         f"down_blocks.{i}.attentions.{j}.transformer_blocks.0.attn{k}.to_out.0",
        #         f"down_blocks.{i}.attentions.{j}.transformer_blocks.0.ff.net.0.proj",
        f"down_blocks.{i}.attentions.{j}.transformer_blocks.0.ff.net.2",
    ]
# mnames += get_all_sub_module_names(pipe.unet.mid_block, "mid_block", lambda x: isinstance(x[1], torch.nn.Linear))
mnames = list(set(linear_moduels) - set(mnames))
print(mnames)


conv_moduels = get_all_sub_module_names(pipe.unet, "", lambda x: isinstance(x[1], torch.nn.Conv2d))
indice = itertools.product(range(0, 4), range(3), range(1, 3))
for i, j, k in indice:
    mnames += [
        # f"up_blocks.{i}.attentions.{j}.proj_in",
        # f"up_blocks.{i}.attentions.{j}.proj_out",
        f"up_blocks.{i}.resnets.{j}.conv{k}",
        f"up_blocks.{i}.resnets.{j}.conv_shortcut",
        f"up_blocks.{i}.upsamplers.{j}.conv",
    ]
    mnames += [
        # f"down_blocks.{i}.attentions.{j}.proj_in",
        # f"down_blocks.{i}.attentions.{j}.proj_out",
        f"down_blocks.{i}.resnets.{j}.conv{k}",
        f"down_blocks.{i}.resnets.{j}.conv_shortcut",
        f"down_blocks.{i}.downsamplers.{j}.conv",
    ]
# mnames += get_all_sub_module_names(pipe.unet.mid_block, "mid_block", lambda x: isinstance(x[1], torch.nn.Linear))
mnames = list(set(conv_moduels) - set(mnames))
print(mnames)


pnames = [mname + "._packed_params._packed_params" for mname in mnames]
default_qconfig = torch.quantization.QConfig(
    # activation=torch.quantization.observer.PerChannelMinMaxObserver.with_args(
    #    dtype=torch.qint8, qscheme=torch.per_channel_affine, ch_axis=0),
    activation=torch.quantization.default_dynamic_quant_observer,
    weight=torch.quantization.observer.MinMaxObserver.with_args(
        # dtype=torch.qint8, qscheme=torch.per_tensor_affine))
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    ),
)
qconfig_spec = dict(zip(mnames, itertools.repeat(default_qconfig)))
# qconfig_spec = dict(zip([torch.nn.Linear], itertools.repeat(default_qconfig)))

mapping = {
    torch.nn.Linear: torch.nn.quantized.dynamic.Linear,
    torch.nn.Conv2d: torch.nn.quantized.dynamic.Conv2d,
}

unet_int8 = torch.quantization.quantize_dynamic(
    pipe.unet, qconfig_spec, dtype=torch.qint8, mapping=mapping  # the original model
)  # the target dtype for quantized weights
unet_fp32 = pipe.unet
pipe.unet = unet_int8

prompt = [prompt] * 1
images = pipe(prompt).images
for i, image in enumerate(images):
    image.save(os.path.join(args.output_dir, f"{prompt[i][-5:]}_{i}.png"))
