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
import tempfile
from collections import OrderedDict

import paddle
import torch
from diffusers import PaintByExamplePipeline as DiffusersPaintByExamplePipeline

# CLIPImageProcessor need paddlenlp latest
from paddlenlp.transformers import CLIPImageProcessor, CLIPVisionConfig
from ppdiffusers import AutoencoderKL
from ppdiffusers import PaintByExamplePipeline as PPDiffusersPaintByExamplePipeline
from ppdiffusers import PNDMScheduler, UNet2DConditionModel
from ppdiffusers.pipelines.paint_by_example.image_encoder import (
    PaintByExampleImageEncoder,
)

paddle.set_device("cpu")


def convert_to_ppdiffusers(vae_or_unet, dtype="float32", prefix=""):
    need_transpose = []
    for k, v in vae_or_unet.named_modules():
        if isinstance(v, torch.nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = OrderedDict()
    for k, v in vae_or_unet.state_dict().items():
        if k not in need_transpose:
            new_vae_or_unet[prefix + k] = v.cpu().numpy().astype(dtype)
        else:
            new_vae_or_unet[prefix + k] = v.t().cpu().numpy().astype(dtype)
    return new_vae_or_unet


def convert_hf_clip_to_ppnlp_clip(clip, dtype="float32"):
    new_model_state = {}
    transformers2ppnlp = {
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
        ".post_layernorm.": ".ln_post.",
    }
    ignore_value = ["position_ids", "mapper"]
    donot_transpose = ["embeddings", "norm", "concept_embeds", "special_care_embeds"]

    for name, value in clip.state_dict().items():
        # step1: ignore position_ids and mapper
        if any(i in name for i in ignore_value):
            continue
        # step2: transpose nn.Linear weight
        if value.ndim == 2 and not any(i in name for i in donot_transpose):
            value = value.t()
        # step3: hf_name -> ppnlp_name mapping
        for hf_name, ppnlp_name in transformers2ppnlp.items():
            name = name.replace(hf_name, ppnlp_name)
        # step4: 0d tensor -> 1d tensor
        if name == "logit_scale":
            value = value.reshape((1,))

        new_model_state[name] = value.cpu().numpy().astype(dtype)

    # convert mapper
    mappersd = convert_to_ppdiffusers(clip.mapper, prefix="mapper.")
    new_model_state.update(mappersd)

    new_config = {
        "image_resolution": clip.config.image_size,
        "vision_layers": clip.config.num_hidden_layers,
        "vision_heads": clip.config.num_attention_heads,
        "vision_embed_dim": clip.config.hidden_size,
        "vision_patch_size": clip.config.patch_size,
        "vision_mlp_ratio": clip.config.intermediate_size // clip.config.hidden_size,
        "vision_hidden_act": clip.config.hidden_act,
        "projection_dim": clip.config.projection_dim,
    }
    return new_model_state, new_config


def check_keys(model, state_dict):
    cls_name = model.__class__.__name__
    missing_keys = []
    mismatched_keys = []
    for k, v in model.state_dict().items():
        if k not in state_dict.keys():
            missing_keys.append(k)
        if list(v.shape) != list(state_dict[k].shape):
            mismatched_keys.append(k)
    if len(missing_keys):
        missing_keys_str = ", ".join(missing_keys)
        print(f"{cls_name} Found missing_keys {missing_keys_str}!")
    if len(mismatched_keys):
        mismatched_keys_str = ", ".join(mismatched_keys)
        print(f"{cls_name} Found mismatched_keys {mismatched_keys_str}!")


def convert_diffusers_paintbyexample_to_ppdiffusers(pretrained_model_name_or_path, output_path=None):
    # 0. load diffusers pipe and convert to ppdiffusers weights format
    diffusers_pipe = DiffusersPaintByExamplePipeline.from_pretrained(
        pretrained_model_name_or_path, use_auth_token=True
    )
    vae_state_dict = convert_to_ppdiffusers(diffusers_pipe.vae)
    unet_state_dict = convert_to_ppdiffusers(diffusers_pipe.unet)
    image_encoder_state_dict, image_encoder_config = convert_hf_clip_to_ppnlp_clip(diffusers_pipe.image_encoder)

    # 1. vae
    pp_vae = AutoencoderKL.from_config(diffusers_pipe.vae.config)
    pp_vae.set_dict(vae_state_dict)
    check_keys(pp_vae, vae_state_dict)
    # 2. unet
    pp_unet = UNet2DConditionModel.from_config(diffusers_pipe.unet.config)
    pp_unet.set_dict(unet_state_dict)
    check_keys(pp_unet, unet_state_dict)

    # 3. image_encoder
    pp_image_encoder = PaintByExampleImageEncoder(CLIPVisionConfig.from_dict(image_encoder_config))
    pp_image_encoder.set_dict(image_encoder_state_dict)
    check_keys(pp_image_encoder, image_encoder_state_dict)
    # 4. scheduler
    pp_scheduler = PNDMScheduler.from_config(diffusers_pipe.scheduler.config)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # 5. feature_extractor
        diffusers_pipe.feature_extractor.save_pretrained(tmpdirname)
        feature_extractor = CLIPImageProcessor.from_pretrained(tmpdirname)

        # 6. create ppdiffusers pipe
        paddle_pipe = PPDiffusersPaintByExamplePipeline(
            vae=pp_vae,
            image_encoder=pp_image_encoder,
            unet=pp_unet,
            scheduler=pp_scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
            requires_safety_checker=False,
        )

        # 6. save_pretrained
        paddle_pipe.save_pretrained(output_path)
    return paddle_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Fantasy-Studio/Paint-by-Example",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./Paint-by-Example",
        help="The model output path.",
    )
    args = parser.parse_args()
    ppdiffusers_pipe = convert_diffusers_paintbyexample_to_ppdiffusers(
        args.pretrained_model_name_or_path, args.output_path
    )
