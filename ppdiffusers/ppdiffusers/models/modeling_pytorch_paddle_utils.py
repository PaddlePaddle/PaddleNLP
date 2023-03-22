# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
""" PyTorch - Paddle general utilities."""
import paddle.nn as nn

#####################
# PyTorch => Paddle #
#####################


def convert_pytorch_state_dict_to_paddle(pt_state_dict, paddle_model: nn.Layer, sub_layer=None):
    # Step 1: Find Linear layer which need transpose weight
    linear_need_transpose = []
    for k, v in paddle_model.named_sublayers(include_self=True):
        if isinstance(v, nn.Linear):
            if sub_layer is not None and sub_layer not in k:
                continue
            linear_need_transpose.append(k + ".weight")

    paddle_state_dict = {}
    ignore_keys = ["position_ids", ".num_batches_tracked"]
    ptname2pdname = {
        # torch.nn.BatchNorm2d -> paddle.nn.BatchNorm2D
        ".running_var": "._variance",
        ".running_mean": "._mean",
    }
    # Need to change some parameters name to match paddle names
    for pt_key, pt_tensor in pt_state_dict.items():
        # only convert sub_layer state dict
        if sub_layer is not None and sub_layer not in pt_key:
            continue
        # (0) ignore_keys
        if any(i in pt_key for i in ignore_keys):
            continue
        # (1) transpose linear
        if pt_key in linear_need_transpose and pt_tensor.ndim == 2:
            pt_tensor = pt_tensor.T
        # (2) 0d tensor -> 1d tensor
        if pt_tensor.ndim == 0:
            pt_tensor = pt_tensor.reshape((1,))
        # (3) name mapping
        for old_key, new_key in ptname2pdname.items():
            pt_key = pt_key.replace(old_key, new_key)

        paddle_state_dict[pt_key] = pt_tensor
    return paddle_state_dict


def convert_paddle_state_dict_to_pytorch(pd_state_dict, paddle_model: nn.Layer):
    # Step 2: Find Linear layer which need transpose weight
    linear_need_transpose = []
    for k, v in paddle_model.named_sublayers(include_self=True):
        if isinstance(v, nn.Linear):
            linear_need_transpose.append(k + ".weight")

    pytorch_state_dict = {}
    ignore_keys = ["position_ids"]
    ptname2pdname = {
        # torch.nn.BatchNorm2d -> paddle.nn.BatchNorm2D
        ".running_var": "._variance",
        ".running_mean": "._mean",
    }
    # Need to change some parameters name to match Flax names
    for pd_key, pd_tensor in pd_state_dict.items():
        # (0) ignore_keys
        if any(i in pd_key for i in ignore_keys):
            continue
        # (1) transpose linear
        if pd_key in linear_need_transpose and pd_tensor.ndim == 2:
            pd_tensor = pd_tensor.T
        # TODO maybe not true
        # (2) 1d tensor -> 0d tensor
        if pd_tensor.ndim == 1:
            pd_tensor = pd_tensor.squeeze()
        # (3) name mapping
        for old_key, new_key in ptname2pdname.items():
            pd_key = pd_key.replace(new_key, old_key)
        if hasattr(paddle_model, "paddle_torch_name_mapping"):
            pd_key = paddle_model.paddle_torch_name_mapping.get(pd_key, pd_key)
        pytorch_state_dict[pd_key] = pd_tensor.contiguous() if hasattr(pd_tensor, "contiguous") else pd_tensor
    return pytorch_state_dict


# if __name__ == "__main__":
#     from paddlenlp.transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionModel, CLIPVisionModelWithProjection, BertModel, DPTForDepthEstimation, BitBackbone
#     from ppdiffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
#     from ppdiffusers.pipelines.stable_diffusion_safe.safety_checker import SafeStableDiffusionSafetyChecker
#     from ppdiffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
#     from ppdiffusers.pipelines.paint_by_example.image_encoder import PaintByExampleImageEncoder

#     clip = [(CLIPTextModel, "runwayml/stable-diffusion-v1-5", "text_encoder"),  # test safetensors
#             (CLIPTextModel, "CompVis/stable-diffusion-v1-4", "text_encoder"),
#             (CLIPTextModelWithProjection, "shi-labs/versatile-diffusion", "text_encoder"),
#             (StableDiffusionSafetyChecker,"CompVis/stable-diffusion-v1-4", "safety_checker"),
#             (SafeStableDiffusionSafetyChecker,"CompVis/stable-diffusion-v1-4", "safety_checker"),
#             (CLIPVisionModelWithProjection, "shi-labs/versatile-diffusion", "image_encoder"),
#             (PaintByExampleImageEncoder, "Fantasy-Studio/Paint-by-Example", "image_encoder"),
#         ]
#     bert = [(BertModel, "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", "text_encoder"),
#             (RobertaSeriesModelWithTransformation, "BAAI/AltDiffusion", "text_encoder")]
#     other = [(DPTForDepthEstimation, "stabilityai/stable-diffusion-2-depth", "depth_estimator")] # test safetensors
#     for cls_, name, subfolder in clip+bert+other:
#         print(name + "======" + subfolder)
#         model, load_info = cls_.from_pretrained(
#             name,
#             output_loading_info=True,
#             subfolder=subfolder,
#             from_hf_hub=True,
#             from_diffusers=True,
#             resume_download=True,
#             cache_dir="nihao",
#         )
