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
import argparse
import os
from collections import OrderedDict

import paddle
import torch
from diffusers import ControlNetModel as DiffusersControlNetModel

from ppdiffusers import ControlNetModel as PPDiffusersControlNetModel

paddle.set_device("cpu")


def convert_to_ppdiffusers(controlnet, dtype="float32"):
    need_transpose = []
    for k, v in controlnet.named_modules():
        if isinstance(v, torch.nn.Linear):
            need_transpose.append(k + ".weight")
    new_controlnet = OrderedDict()
    for k, v in controlnet.state_dict().items():
        if k not in need_transpose:
            new_controlnet[k] = v.cpu().numpy().astype(dtype)
        else:
            new_controlnet[k] = v.t().cpu().numpy().astype(dtype)
    return new_controlnet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lllyasviel/sd-controlnet-canny",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="paddle_models/sd-controlnet-canny",
        help="The output path.",
    )
    args = parser.parse_args()

    th_controlnet = DiffusersControlNetModel.from_pretrained(args.pretrained_model_name_or_path)
    controlnet_state_dict = convert_to_ppdiffusers(th_controlnet)
    pp_controlnet = PPDiffusersControlNetModel.from_config(th_controlnet.config)
    pp_controlnet.set_dict(controlnet_state_dict)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    pp_controlnet.save_pretrained(args.output_path)
