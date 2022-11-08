# coding=utf-8
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
""" Conversion script for the LDM checkpoints. """

import argparse
import os

import torch
import paddle

paddle.set_device("cpu")

try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(
        "OmegaConf is required to convert the LDM checkpoints. Please install it with `pip install OmegaConf`."
    )


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "query.weight")
        new_item = new_item.replace("q.bias", "query.bias")

        new_item = new_item.replace("k.weight", "key.weight")
        new_item = new_item.replace("k.bias", "key.bias")

        new_item = new_item.replace("v.weight", "value.weight")
        new_item = new_item.replace("v.bias", "value.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(paths,
                         checkpoint,
                         old_checkpoint,
                         attention_paths_to_split=None,
                         additional_replacements=None,
                         config=None):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.
    Assigns the weights to the new checkpoint.
    """
    assert isinstance(
        paths, list
    ), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1,
                            channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels //
                                             num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"],
                                            replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params

    block_out_channels = [
        unet_params.model_channels * mult for mult in unet_params.channel_mult
    ]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=unet_params.image_size,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params.num_res_blocks,
        cross_attention_dim=unet_params.context_dim,
        attention_head_dim=unet_params.num_heads,
    )

    return config


def create_vae_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.first_stage_config.params.ddconfig
    _ = original_config.model.params.first_stage_config.params.embed_dim

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=vae_params.resolution,
        in_channels=vae_params.in_channels,
        out_channels=vae_params.out_ch,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=vae_params.z_channels,
        layers_per_block=vae_params.num_res_blocks,
    )
    return config


def convert_ldm_unet_checkpoint(unet_state_dict, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict[
        "time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict[
        "time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict[
        "time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict[
        "time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict[
        "input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({
        ".".join(layer.split(".")[:2])
        for layer in unet_state_dict if "input_blocks" in layer
    })
    input_blocks = {
        layer_id:
        [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({
        ".".join(layer.split(".")[:2])
        for layer in unet_state_dict if "middle_block" in layer
    })
    middle_blocks = {
        layer_id:
        [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({
        ".".join(layer.split(".")[:2])
        for layer in unet_state_dict if "output_blocks" in layer
    })
    output_blocks = {
        layer_id:
        [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key
            and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [
            key for key in input_blocks[i] if f"input_blocks.{i}.1" in key
        ]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[
                f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                    f"input_blocks.{i}.0.op.weight")
            new_checkpoint[
                f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                    f"input_blocks.{i}.0.op.bias")

        paths = renew_resnet_paths(resnets)
        meta_path = {
            "old": f"input_blocks.{i}.0",
            "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"
        }
        assign_to_checkpoint(paths,
                             new_checkpoint,
                             unet_state_dict,
                             additional_replacements=[meta_path],
                             config=config)

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {
                "old": f"input_blocks.{i}.1",
                "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"
            }
            assign_to_checkpoint(paths,
                                 new_checkpoint,
                                 unet_state_dict,
                                 additional_replacements=[meta_path],
                                 config=config)

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths,
                         new_checkpoint,
                         unet_state_dict,
                         config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths,
                         new_checkpoint,
                         unet_state_dict,
                         config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(attentions_paths,
                         new_checkpoint,
                         unet_state_dict,
                         additional_replacements=[meta_path],
                         config=config)

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [
            shave_segments(name, 2) for name in output_blocks[i]
        ]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [
                key for key in output_blocks[i] if f"output_blocks.{i}.0" in key
            ]
            attentions = [
                key for key in output_blocks[i] if f"output_blocks.{i}.1" in key
            ]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {
                "old": f"output_blocks.{i}.0",
                "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"
            }
            assign_to_checkpoint(paths,
                                 new_checkpoint,
                                 unet_state_dict,
                                 additional_replacements=[meta_path],
                                 config=config)

            if ["conv.weight", "conv.bias"] in output_block_list.values():
                index = list(output_block_list.values()).index(
                    ["conv.weight", "conv.bias"])
                new_checkpoint[
                    f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                        f"output_blocks.{i}.{index}.conv.weight"]
                new_checkpoint[
                    f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                        f"output_blocks.{i}.{index}.conv.bias"]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new":
                    f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(paths,
                                     new_checkpoint,
                                     unet_state_dict,
                                     additional_replacements=[meta_path],
                                     config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers,
                                                n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join([
                    "up_blocks",
                    str(block_id), "resnets",
                    str(layer_in_block_id), path["new"]
                ])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


def convert_ldm_vae_checkpoint(vae_state_dict, config):

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict[
        "encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict[
        "encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict[
        "encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict[
        "encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict[
        "encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict[
        "encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict[
        "decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict[
        "decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict[
        "decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict[
        "decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict[
        "decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict[
        "decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict[
        "post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict[
        "post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({
        ".".join(layer.split(".")[:3])
        for layer in vae_state_dict if "encoder.down" in layer
    })
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
        for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({
        ".".join(layer.split(".")[:3])
        for layer in vae_state_dict if "decoder.up" in layer
    })
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
        for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [
            key for key in down_blocks[i]
            if f"down.{i}" in key and f"down.{i}.downsample" not in key
        ]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                    f"encoder.down.{i}.downsample.conv.weight")
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                    f"encoder.down.{i}.downsample.conv.bias")

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {
            "old": f"down.{i}.block",
            "new": f"down_blocks.{i}.resnets"
        }
        assign_to_checkpoint(paths,
                             new_checkpoint,
                             vae_state_dict,
                             additional_replacements=[meta_path],
                             config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [
            key for key in mid_resnets if f"encoder.mid.block_{i}" in key
        ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {
            "old": f"mid.block_{i}",
            "new": f"mid_block.resnets.{i - 1}"
        }
        assign_to_checkpoint(paths,
                             new_checkpoint,
                             vae_state_dict,
                             additional_replacements=[meta_path],
                             config=config)

    mid_attentions = [
        key for key in vae_state_dict if "encoder.mid.attn" in key
    ]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths,
                         new_checkpoint,
                         vae_state_dict,
                         additional_replacements=[meta_path],
                         config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id]
            if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                    f"decoder.up.{block_id}.upsample.conv.weight"]
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                    f"decoder.up.{block_id}.upsample.conv.bias"]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {
            "old": f"up.{block_id}.block",
            "new": f"up_blocks.{i}.resnets"
        }
        assign_to_checkpoint(paths,
                             new_checkpoint,
                             vae_state_dict,
                             additional_replacements=[meta_path],
                             config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [
            key for key in mid_resnets if f"decoder.mid.block_{i}" in key
        ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {
            "old": f"mid.block_{i}",
            "new": f"mid_block.resnets.{i - 1}"
        }
        assign_to_checkpoint(paths,
                             new_checkpoint,
                             vae_state_dict,
                             additional_replacements=[meta_path],
                             config=config)

    mid_attentions = [
        key for key in vae_state_dict if "decoder.mid.attn" in key
    ]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths,
                         new_checkpoint,
                         vae_state_dict,
                         additional_replacements=[meta_path],
                         config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def convert_unet(unet, out_path):
    need_transpose = [
        'time_embedding.linear_1.weight', 'time_embedding.linear_2.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.weight',
        'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.weight',
        'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.weight',
        'down_blocks.0.resnets.0.time_emb_proj.weight',
        'down_blocks.0.resnets.1.time_emb_proj.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.weight',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.weight',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.weight',
        'down_blocks.1.resnets.0.time_emb_proj.weight',
        'down_blocks.1.resnets.1.time_emb_proj.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.weight',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.weight',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.weight',
        'down_blocks.2.resnets.0.time_emb_proj.weight',
        'down_blocks.2.resnets.1.time_emb_proj.weight',
        'down_blocks.3.resnets.0.time_emb_proj.weight',
        'down_blocks.3.resnets.1.time_emb_proj.weight',
        'up_blocks.0.resnets.0.time_emb_proj.weight',
        'up_blocks.0.resnets.1.time_emb_proj.weight',
        'up_blocks.0.resnets.2.time_emb_proj.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.1.resnets.0.time_emb_proj.weight',
        'up_blocks.1.resnets.1.time_emb_proj.weight',
        'up_blocks.1.resnets.2.time_emb_proj.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.2.resnets.0.time_emb_proj.weight',
        'up_blocks.2.resnets.1.time_emb_proj.weight',
        'up_blocks.2.resnets.2.time_emb_proj.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v.weight',
        'up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.weight',
        'up_blocks.3.resnets.0.time_emb_proj.weight',
        'up_blocks.3.resnets.1.time_emb_proj.weight',
        'up_blocks.3.resnets.2.time_emb_proj.weight',
        'mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight',
        'mid_block.attentions.0.transformer_blocks.0.attn1.to_k.weight',
        'mid_block.attentions.0.transformer_blocks.0.attn1.to_v.weight',
        'mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.weight',
        'mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight',
        'mid_block.attentions.0.transformer_blocks.0.ff.net.2.weight',
        'mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight',
        'mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight',
        'mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight',
        'mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight',
        'mid_block.resnets.0.time_emb_proj.weight',
        'mid_block.resnets.1.time_emb_proj.weight'
    ]
    new_unet = {}
    for k, v in unet.items():
        if k not in need_transpose:
            new_unet[k] = v.numpy()
        else:
            new_unet[k] = v.t().numpy()

    paddle.save(new_unet, out_path)

    return new_unet


def convert_vae(vae, out_path):
    need_transpose = [
        'encoder.mid_block.attentions.0.query.weight',
        'encoder.mid_block.attentions.0.key.weight',
        'encoder.mid_block.attentions.0.value.weight',
        'encoder.mid_block.attentions.0.proj_attn.weight',
        'decoder.mid_block.attentions.0.query.weight',
        'decoder.mid_block.attentions.0.key.weight',
        'decoder.mid_block.attentions.0.value.weight',
        'decoder.mid_block.attentions.0.proj_attn.weight'
    ]
    new_vae = {}
    for k, v in vae.items():
        if k not in need_transpose:
            new_vae[k] = v.numpy()
        else:
            new_vae[k] = v.t().numpy()
    paddle.save(new_vae, out_path)
    return new_vae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_checkpoint_path",
                        default="init_weights/unet.pt",
                        type=str,
                        required=False,
                        help="Path to the checkpoint to convert.")
    parser.add_argument("--vae_checkpoint_path",
                        default="init_weights/vae.pt",
                        type=str,
                        required=False,
                        help="Path to the checkpoint to convert.")
    parser.add_argument(
        "--original_config_file",
        default="configs/latent-diffusion/text2img_L12H768_unet800M.yaml",
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument("--dump_path",
                        default="pretrained_paddle_model",
                        type=str,
                        help="Path to the output model.")

    args = parser.parse_args()

    original_config = OmegaConf.load(args.original_config_file)
    unet_checkpoint = torch.load(args.unet_checkpoint_path, map_location="cpu")
    vae_checkpoint = torch.load(args.vae_checkpoint_path, map_location="cpu")

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        unet_checkpoint, unet_config)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(
        vae_checkpoint, vae_config)
    os.makedirs(os.path.join(args.dump_path, "unet"), exist_ok=True)
    os.makedirs(os.path.join(args.dump_path, "vae"), exist_ok=True)

    convert_unet(converted_unet_checkpoint,
                 os.path.join(args.dump_path, "unet", "model_state.pdparams"))
    convert_vae(converted_vae_checkpoint,
                os.path.join(args.dump_path, "vae", "model_state.pdparams"))
