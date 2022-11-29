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
from diffusers import LDMTextToImagePipeline as DiffusersLDMTextToImagePipeline

from paddlenlp.transformers import BertTokenizer
from ppdiffusers import AutoencoderKL, DDIMScheduler, LDMBertModel
from ppdiffusers import LDMTextToImagePipeline as PPDiffusersLDMTextToImagePipeline
from ppdiffusers import LMSDiscreteScheduler, PNDMScheduler, UNet2DConditionModel

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


def convert_hf_ldmbert_to_ppnlp_ldmbert(ldmbert, dtype="float32"):
    transformers2ppnlp = {
        "model.embed_tokens.weight": "embeddings.word_embeddings.weight",
        "model.embed_positions.weight": "embeddings.position_embeddings.weight",
        "model.layer_norm.": "final_layer_norm.",
        "model.layers": "encoder.layers",
        ".self_attn_layer_norm.": ".norm1.",
        ".final_layer_norm.": ".norm2.",
        ".fc1.": ".linear1.",
        ".fc2.": ".linear2.",
    }
    ignore_value = ["to_logits"]
    donot_transpose = ["embed_tokens", "embed_positions", "norm"]
    new_model_state = OrderedDict()
    for name, value in ldmbert.state_dict().items():
        # step1: ignore to_logits
        if any(i in name for i in ignore_value):
            continue
        # step2: transpose nn.Linear weight
        if value.ndim == 2 and not any(i in name for i in donot_transpose):
            value = value.t()
        # step3: hf_name -> ppnlp_name mapping
        for hf_name, ppnlp_name in transformers2ppnlp.items():
            name = name.replace(hf_name, ppnlp_name)
        new_model_state[name] = value.cpu().numpy().astype(dtype)

    new_config = {
        "vocab_size": ldmbert.config.vocab_size,
        "max_position_embeddings": ldmbert.config.max_position_embeddings,
        "encoder_layers": ldmbert.config.encoder_layers,
        "encoder_ffn_dim": ldmbert.config.encoder_ffn_dim,
        "encoder_attention_heads": ldmbert.config.encoder_attention_heads,
        "head_dim": ldmbert.config.head_dim,
        "activation_function": ldmbert.config.activation_function,
        "d_model": ldmbert.config.d_model,
        "dropout": 0.0,  # we do not use dropout in original ldmbert
        "attention_dropout": ldmbert.config.attention_dropout,
        "activation_dropout": ldmbert.config.activation_dropout,
        "init_std": ldmbert.config.init_std,
        "pad_token_id": ldmbert.config.pad_token_id,
    }
    return new_model_state, new_config


def convert_diffusers_stable_diffusion_to_ppdiffusers(pretrained_model_name_or_path, output_path=None):
    # 0. load diffusers pipe and convert to ppdiffusers weights format
    diffusers_pipe = DiffusersLDMTextToImagePipeline.from_pretrained(
        pretrained_model_name_or_path, use_auth_token=True
    )
    vqvae_state_dict = convert_to_ppdiffusers(diffusers_pipe.vqvae)
    unet_state_dict = convert_to_ppdiffusers(diffusers_pipe.unet)
    bert_state_dict, bert_config = convert_hf_ldmbert_to_ppnlp_ldmbert(diffusers_pipe.bert)

    # 1. vqvae
    pp_vqvae = AutoencoderKL(**diffusers_pipe.vqvae.config)
    pp_vqvae.set_dict(vqvae_state_dict)

    # 2. unet
    pp_unet = UNet2DConditionModel(**diffusers_pipe.unet.config)
    pp_unet.set_dict(unet_state_dict)

    # 3. bert
    pp_bert = LDMBertModel(**bert_config)
    pp_bert.set_dict(bert_state_dict)

    # 4. scheduler
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
        # 5. tokenizer
        diffusers_pipe.tokenizer.save_pretrained(tmpdirname)
        pp_tokenizer = BertTokenizer.from_pretrained(tmpdirname, model_max_length=77)

        # 6. create ppdiffusers pipe
        paddle_pipe = PPDiffusersLDMTextToImagePipeline(
            vqvae=pp_vqvae, bert=pp_bert, tokenizer=pp_tokenizer, unet=pp_unet, scheduler=pp_scheduler
        )

        # 7. save_pretrained
        paddle_pipe.save_pretrained(output_path)
    return paddle_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/ldm-text2im-large-256",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ldm-text2im-large-256-ppdiffusers",
        help="The model output path.",
    )
    args = parser.parse_args()
    ppdiffusers_pipe = convert_diffusers_stable_diffusion_to_ppdiffusers(
        args.pretrained_model_name_or_path, args.output_path
    )
