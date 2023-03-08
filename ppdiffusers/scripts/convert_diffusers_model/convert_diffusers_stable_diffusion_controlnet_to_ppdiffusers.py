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
    StableDiffusionControlNetPipeline as DiffusersStableDiffusionControlNetPipeline,
)

from paddlenlp.transformers import (
    CLIPFeatureExtractor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
)
from ppdiffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ppdiffusers import (
    StableDiffusionControlNetPipeline as PPDiffusersStableDiffusionControlNetPipeline,
)
from ppdiffusers import UNet2DConditionModel
from ppdiffusers.configuration_utils import FrozenDict
from ppdiffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

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


def convert_diffusers_stable_diffusion_controlnet_to_ppdiffusers(pretrained_model_name_or_path, output_path=None):
    # 0. load diffusers pipe and convert to ppdiffusers weights format
    diffusers_pipe = DiffusersStableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path, use_auth_token=True
    )
    requires_safety_checker = getattr(diffusers_pipe, "requires_safety_checker", False)
    vae_state_dict = convert_to_ppdiffusers(diffusers_pipe.vae)
    unet_state_dict = convert_to_ppdiffusers(diffusers_pipe.unet)
    controlnet_state_dict = convert_to_ppdiffusers(diffusers_pipe.controlnet)
    text_encoder_state_dict, text_encoder_config = convert_hf_clip_to_ppnlp_clip(
        diffusers_pipe.text_encoder, is_text_encoder=True
    )

    # 1. vae
    pp_vae = AutoencoderKL.from_config(diffusers_pipe.vae.config)

    pp_vae.set_dict(vae_state_dict)

    # 2. unet
    pp_unet = UNet2DConditionModel.from_config(diffusers_pipe.unet.config)

    pp_unet.set_dict(unet_state_dict)

    # 3. controlnet
    pp_controlnet = ControlNetModel.from_config(diffusers_pipe.controlnet.config)

    pp_controlnet.set_dict(controlnet_state_dict)

    # 4. text_encoder
    pp_text_encoder = CLIPTextModel(CLIPTextConfig.from_dict(text_encoder_config))
    pp_text_encoder.set_dict(text_encoder_state_dict)

    # 5. scheduler
    beta_start = diffusers_pipe.scheduler.beta_start
    beta_end = diffusers_pipe.scheduler.beta_end
    scheduler_type = diffusers_pipe.scheduler._class_name.lower()
    if "pndm" in scheduler_type:
        pp_scheduler = PNDMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            # Make sure the scheduler compatible with DDIM
            set_alpha_to_one=False,
            steps_offset=1,
            # Make sure the scheduler compatible with PNDM
            skip_prk_steps=True,
        )
    elif "lms" in scheduler_type:
        pp_scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear")
    elif "ddim" in scheduler_type:
        pp_scheduler = DDIMScheduler(
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

        if requires_safety_checker:
            # 7. feature_extractor
            # diffusers_pipe.feature_extractor.save_pretrained(tmpdirname)
            pp_feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-v1-4/feature_extractor"
            )
            # 8. safety_checker
            safety_checker_state_dict, safety_checker_config = convert_hf_clip_to_ppnlp_clip(
                diffusers_pipe.safety_checker, is_text_encoder=False
            )
            pp_safety_checker = StableDiffusionSafetyChecker(CLIPVisionConfig.from_dict(safety_checker_config))
            pp_safety_checker.set_dict(safety_checker_state_dict)
            # 9. create ppdiffusers pipe
            paddle_pipe = PPDiffusersStableDiffusionControlNetPipeline(
                vae=pp_vae,
                text_encoder=pp_text_encoder,
                tokenizer=pp_tokenizer,
                unet=pp_unet,
                controlnet=pp_controlnet,
                safety_checker=pp_safety_checker,
                feature_extractor=pp_feature_extractor,
                scheduler=pp_scheduler,
            )
        else:
            # 9. create ppdiffusers pipe
            paddle_pipe = PPDiffusersStableDiffusionControlNetPipeline(
                vae=pp_vae,
                text_encoder=pp_text_encoder,
                tokenizer=pp_tokenizer,
                unet=pp_unet,
                controlnet=pp_controlnet,
                safety_checker=None,
                feature_extractor=None,
                scheduler=pp_scheduler,
                requires_safety_checker=False,
            )
        if "runwayml/stable-diffusion-inpainting" in pretrained_model_name_or_path:
            _internal_dict = dict(paddle_pipe._internal_dict)
            if _internal_dict["_ppdiffusers_version"] == "0.0.0":
                _internal_dict.update({"_ppdiffusers_version": "0.6.0"})
            paddle_pipe._internal_dict = FrozenDict(_internal_dict)
        # 10. save_pretrained
        paddle_pipe.save_pretrained(output_path)
    return paddle_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="takuma104/control_sd15_canny",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="control_sd15_canny-ppdiffusers",
        help="The model output path.",
    )
    args = parser.parse_args()
    ppdiffusers_pipe = convert_diffusers_stable_diffusion_controlnet_to_ppdiffusers(
        args.pretrained_model_name_or_path, args.output_path
    )
