# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team.
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

# Script for converting a PPDiffusers saved pipeline to a Latent Diffusion Model checkpoint.
# *Only* converts the UNet, VAE, and LDMBert(Text Encoder).
# Does not convert optimizer state or any other thing.

import argparse

import paddle
import torch

from ppdiffusers import LDMTextToImagePipeline

# =================#
# UNet Conversion #
# =================#
paddle.set_device("cpu")

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3-i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i+1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("proj_out.", "proj_attn."),
]


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                print(f"Reshaping {k} for SD format")
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#
# pretty much a no-op


def convert_ppdiffusers_vae_unet_to_diffusers(vae_or_unet, ppdiffusers_vae_unet_checkpoint):
    need_transpose = []
    for k, v in vae_or_unet.named_sublayers(include_self=True):
        if isinstance(v, paddle.nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = {}
    for k, v in ppdiffusers_vae_unet_checkpoint.items():
        if k not in need_transpose:
            new_vae_or_unet[k] = torch.from_numpy(v.numpy())
        else:
            new_vae_or_unet[k] = torch.from_numpy(v.t().numpy())
    return new_vae_or_unet


def convert_ldmbert_state_dict(ldmbert_state_dict, num_layers=32):
    ppdiffusers_mapping_to_orig = {}
    ppdiffusers_mapping_to_orig["embeddings.word_embeddings.weight"] = "cond_stage_model.transformer.token_emb.weight"
    ppdiffusers_mapping_to_orig[
        "embeddings.position_embeddings.weight"
    ] = "cond_stage_model.transformer.pos_emb.emb.weight"
    for i in range(num_layers):
        double_i = 2 * i
        double_i_plus1 = 2 * i + 1
        ppdiffusers_mapping_to_orig[
            f"encoder.layers.{i}.norm1.weight"
        ] = f"cond_stage_model.transformer.attn_layers.layers.{double_i}.0.weight"
        ppdiffusers_mapping_to_orig[
            f"encoder.layers.{i}.norm1.bias"
        ] = f"cond_stage_model.transformer.attn_layers.layers.{double_i}.0.bias"

        ppdiffusers_mapping_to_orig[f"encoder.layers.{i}.self_attn.q_proj.weight"] = (
            f"cond_stage_model.transformer.attn_layers.layers.{double_i}.1.to_q.weight",
            "transpose",
        )
        ppdiffusers_mapping_to_orig[f"encoder.layers.{i}.self_attn.k_proj.weight"] = (
            f"cond_stage_model.transformer.attn_layers.layers.{double_i}.1.to_k.weight",
            "transpose",
        )
        ppdiffusers_mapping_to_orig[f"encoder.layers.{i}.self_attn.v_proj.weight"] = (
            f"cond_stage_model.transformer.attn_layers.layers.{double_i}.1.to_v.weight",
            "transpose",
        )
        ppdiffusers_mapping_to_orig[f"encoder.layers.{i}.self_attn.out_proj.weight"] = (
            f"cond_stage_model.transformer.attn_layers.layers.{double_i}.1.to_out.weight",
            "transpose",
        )
        ppdiffusers_mapping_to_orig[
            f"encoder.layers.{i}.self_attn.out_proj.bias"
        ] = f"cond_stage_model.transformer.attn_layers.layers.{double_i}.1.to_out.bias"

        ppdiffusers_mapping_to_orig[
            f"encoder.layers.{i}.norm2.weight"
        ] = f"cond_stage_model.transformer.attn_layers.layers.{double_i_plus1}.0.weight"
        ppdiffusers_mapping_to_orig[
            f"encoder.layers.{i}.norm2.bias"
        ] = f"cond_stage_model.transformer.attn_layers.layers.{double_i_plus1}.0.bias"
        ppdiffusers_mapping_to_orig[f"encoder.layers.{i}.linear1.weight"] = (
            f"cond_stage_model.transformer.attn_layers.layers.{double_i_plus1}.1.net.0.0.weight",
            "transpose",
        )
        ppdiffusers_mapping_to_orig[
            f"encoder.layers.{i}.linear1.bias"
        ] = f"cond_stage_model.transformer.attn_layers.layers.{double_i_plus1}.1.net.0.0.bias"
        ppdiffusers_mapping_to_orig[f"encoder.layers.{i}.linear2.weight"] = (
            f"cond_stage_model.transformer.attn_layers.layers.{double_i_plus1}.1.net.2.weight",
            "transpose",
        )
        ppdiffusers_mapping_to_orig[
            f"encoder.layers.{i}.linear2.bias"
        ] = f"cond_stage_model.transformer.attn_layers.layers.{double_i_plus1}.1.net.2.bias"

    ppdiffusers_mapping_to_orig["final_layer_norm.weight"] = "cond_stage_model.transformer.norm.weight"
    ppdiffusers_mapping_to_orig["final_layer_norm.bias"] = "cond_stage_model.transformer.norm.bias"

    new_state_dict = {}
    for k, v in ldmbert_state_dict.items():
        new_name = ppdiffusers_mapping_to_orig[k]
        need_transpose = False
        if isinstance(new_name, (list, tuple)):
            need_transpose = True
            new_name = new_name[0]
        new_state_dict[new_name] = torch.from_numpy(v.t().numpy()) if need_transpose else torch.from_numpy(v.numpy())

    # dummpy weights, we donot use this!
    new_state_dict["cond_stage_model.transformer.to_logits.weight"] = torch.zeros(
        new_state_dict["cond_stage_model.transformer.token_emb.weight"].shape
    )
    new_state_dict["cond_stage_model.transformer.to_logits.bias"] = torch.zeros(
        new_state_dict["cond_stage_model.transformer.token_emb.weight"].shape[0]
    )
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", default=None, type=str, required=True, help="Path to the model to convert."
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")

    args = parser.parse_args()
    pipe = LDMTextToImagePipeline.from_pretrained(args.model_name_or_path)

    # Convert the UNet model
    unet_state_dict = convert_ppdiffusers_vae_unet_to_diffusers(pipe.unet, pipe.unet.state_dict())
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

    # Convert the VAE model
    vae_state_dict = convert_ppdiffusers_vae_unet_to_diffusers(pipe.vqvae, pipe.vqvae.state_dict())
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    # Convert the ldmbert model
    text_enc_dict = convert_ldmbert_state_dict(pipe.bert.state_dict(), num_layers=pipe.bert.config["encoder_layers"])

    # Put together new checkpoint
    state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}
    state_dict = {"state_dict": state_dict}
    torch.save(state_dict, args.dump_path)
