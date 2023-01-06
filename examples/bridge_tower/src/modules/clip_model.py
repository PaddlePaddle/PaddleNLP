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

import os

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.transformers import CLIPVisionModel


def adapt_position_encoding(model, patch_size=32, after=384, suffix="vision_model.positional_embedding.weight"):
    keys = [k for k in model if k.endswith(suffix)]
    assert len(keys) == 1
    key = keys[0]
    origin_pos_embed = model[key]
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    # breakpoint()
    grid_before = int(np.sqrt(origin_pos_embed.shape[1] - 1))
    before = int(grid_before * patch_size)
    assert (before % patch_size) == 0
    grid_after = after // patch_size
    assert (after % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = F.interpolate(pos_embed.transpose((2, 0, 1)).unsqueeze(0), size=new_size, mode="bicubic")
    pos_embed = pos_embed.squeeze(0).transpose((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = paddle.concat((origin_pos_embed[0, 0:1, :], pos_embed), axis=0).unsqueeze(0)
    assert pos_embed.shape == [1, grid_after * grid_after + 1, embed_dim]
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    model[key] = pos_embed
    return model


def build_model(
    name="openai/clip-vit-base-patch16",
    resolution_after=576,
    model_type="BT",
    vit_layernorm_shared=True,
    vit_remove_last=False,
):
    # 采用了硬编码的方式加载了预训练模型，主要用于计算adapt_position_encoding的参数
    checkpoint_path = "checkpoints"
    model_path = os.path.join(checkpoint_path, "model_state.pdparams")
    if not os.path.exists(model_path):
        from paddlenlp.transformers import CLIPModel

        model = CLIPModel.from_pretrained(name)
        model.save_pretrained(checkpoint_path)

    model_state = paddle.load(model_path)
    # resolution_after会变化，初始化的可能不是默认的模型
    print(name)
    model = CLIPVisionModel.from_pretrained(name, image_resolution=resolution_after)
    # Load original state dict
    state_dict = model_state

    vision_patch_size = state_dict["vision_model.conv1.weight"].shape[-1]
    grid_size = round((state_dict["vision_model.positional_embedding.weight"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    model_dict = model.state_dict()
    pretrained_dict = state_dict
    # reinitialize positional embedding
    if resolution_after != image_resolution:
        pretrained_dict = adapt_position_encoding(
            pretrained_dict, after=resolution_after, patch_size=vision_patch_size
        )

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # breakpoint()
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_dict(model_dict)
    return model
