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
from diffusers import (
    StableDiffusionUpscalePipeline as DiffusersStableDiffusionUpscalePipeline,
)

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ppdiffusers import (
    StableDiffusionUpscalePipeline as PPDiffusersStableDiffusionUpscalePipeline,
)
from ppdiffusers import UNet2DConditionModel

paddle.set_device("cpu")


def convert_to_ppdiffusers(vae_or_unet, dtype="float32"):
    need_transpose = []
    for k, v in vae_or_unet.named_modules():
        if isinstance(v, torch.nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = OrderedDict()
    for k, v in vae_or_unet.state_dict().items():
        if k not in need_transpose:
            new_vae_or_unet[k] = v.cpu().numpy().astype(dtype)
        else:
            new_vae_or_unet[k] = v.t().cpu().numpy().astype(dtype)
    return new_vae_or_unet


def convert_hf_clip_to_ppnlp_clip(clip, dtype="float32", is_text_encoder=True):
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
        ".vision_model.": ".",
    }
    ignore_value = ["position_ids"]
    donot_transpose = ["embeddings", "norm", "concept_embeds", "special_care_embeds"]

    for name, value in clip.state_dict().items():
        # step1: ignore position_ids
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
        # step5: safety_checker need prefix "clip."
        if "vision_model" in name:
            name = "clip." + name
        new_model_state[name] = value.cpu().numpy().astype(dtype)

    if is_text_encoder:
        new_config = {
            "max_text_length": clip.config.max_position_embeddings,
            "vocab_size": clip.config.vocab_size,
            "text_embed_dim": clip.config.hidden_size,
            "text_heads": clip.config.num_attention_heads,
            "text_layers": clip.config.num_hidden_layers,
            "text_hidden_act": clip.config.hidden_act,
            "projection_dim": clip.config.projection_dim,
            "initializer_range": clip.config.initializer_range,
            "initializer_factor": clip.config.initializer_factor,
        }
    else:
        new_config = {
            "image_resolution": clip.config.vision_config.image_size,
            "vision_layers": clip.config.vision_config.num_hidden_layers,
            "vision_heads": clip.config.vision_config.num_attention_heads,
            "vision_embed_dim": clip.config.vision_config.hidden_size,
            "vision_patch_size": clip.config.vision_config.patch_size,
            "vision_mlp_ratio": clip.config.vision_config.intermediate_size // clip.config.vision_config.hidden_size,
            "vision_hidden_act": clip.config.vision_config.hidden_act,
            "projection_dim": clip.config.projection_dim,
        }
    return new_model_state, new_config


def convert_diffusers_stable_diffusion_to_ppdiffusers(pretrained_model_name_or_path, output_path=None):
    # 0. load diffusers pipe and convert to ppdiffusers weights format
    diffusers_pipe = DiffusersStableDiffusionUpscalePipeline.from_pretrained(
        pretrained_model_name_or_path, use_auth_token=True
    )
    vae_state_dict = convert_to_ppdiffusers(diffusers_pipe.vae)
    unet_state_dict = convert_to_ppdiffusers(diffusers_pipe.unet)
    text_encoder_state_dict, text_encoder_config = convert_hf_clip_to_ppnlp_clip(
        diffusers_pipe.text_encoder, is_text_encoder=True
    )
    max_noise_level = diffusers_pipe.max_noise_level

    # 1. vae
    pp_vae = AutoencoderKL(**diffusers_pipe.vae.config)
    pp_vae.set_dict(vae_state_dict)

    # 2. unet
    pp_unet = UNet2DConditionModel(**diffusers_pipe.unet.config)
    pp_unet.set_dict(unet_state_dict)

    # 3. text_encoder
    pp_text_encoder = CLIPTextModel(**text_encoder_config)
    pp_text_encoder.set_dict(text_encoder_state_dict)

    # 4. scheduler
    beta_start = diffusers_pipe.scheduler.beta_start
    beta_end = diffusers_pipe.scheduler.beta_end
    beta_schedule = diffusers_pipe.scheduler.beta_schedule
    num_train_timesteps = diffusers_pipe.scheduler.num_train_timesteps
    scheduler_type = diffusers_pipe.scheduler._class_name.lower()
    if "pndm" in scheduler_type:
        pp_scheduler = PNDMScheduler(
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            skip_prk_steps=True,
        )
    elif "lms" in scheduler_type:
        pp_scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule)
    elif "ddim" in scheduler_type:
        pp_scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=False,
            prediction_type="v_prediction",
            set_alpha_to_one=False,
            steps_offset=1,
        )
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    # 5. low_res_scheduler
    beta_start = diffusers_pipe.low_res_scheduler.beta_start
    beta_end = diffusers_pipe.low_res_scheduler.beta_end
    num_train_timesteps = diffusers_pipe.low_res_scheduler.num_train_timesteps
    beta_schedule = diffusers_pipe.low_res_scheduler.beta_schedule
    scheduler_type = diffusers_pipe.low_res_scheduler._class_name.lower()
    if "pndm" in scheduler_type:
        pp_low_res_scheduler = PNDMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            # Make sure the scheduler compatible with DDIM
            set_alpha_to_one=False,
            steps_offset=1,
            # Make sure the scheduler compatible with PNDM
            skip_prk_steps=True,
        )
    elif "ddpm" in scheduler_type:
        pp_low_res_scheduler = DDPMScheduler(
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
        )
    elif "lms" in scheduler_type:
        pp_low_res_scheduler = LMSDiscreteScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule
        )
    elif "ddim" in scheduler_type:
        pp_low_res_scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            # Make sure the scheduler compatible with DDIM
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    with tempfile.TemporaryDirectory() as tmpdirname:
        # 6. tokenizer
        diffusers_pipe.tokenizer.save_pretrained(tmpdirname)
        pp_tokenizer = CLIPTokenizer.from_pretrained(tmpdirname)
        # 7. create ppdiffusers pipe
        paddle_pipe = PPDiffusersStableDiffusionUpscalePipeline(
            max_noise_level=max_noise_level,
            vae=pp_vae,
            text_encoder=pp_text_encoder,
            tokenizer=pp_tokenizer,
            unet=pp_unet,
            low_res_scheduler=pp_low_res_scheduler,
            scheduler=pp_scheduler,
        )

        # 9. save_pretrained
        paddle_pipe.save_pretrained(output_path)
    return paddle_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-x4-upscaler",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="stable-diffusion-x4-upscaler-ppdiffusers",
        help="The model output path.",
    )
    args = parser.parse_args()
    ppdiffusers_pipe = convert_diffusers_stable_diffusion_to_ppdiffusers(
        args.pretrained_model_name_or_path, args.output_path
    )
