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
""" PyTorch - Paddle general utilities."""
import re
from .utils import logging

logger = logging.get_logger(__name__)


def rename_key(key):
    regex = r"\w+[.]\d+"
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key


#####################
# PyTorch => Paddle #
#####################


def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor,
                                  random_paddle_state_dict):
    """Rename PT weight names to corresponding Paddle weight names and reshape tensor if necessary"""

    # conv norm or layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias", )
    if (any("norm" in str_ for str_ in pt_tuple_key)
            and (pt_tuple_key[-1] in ["bias", "beta"])
            and (pt_tuple_key[:-1] + ("bias", ) in random_paddle_state_dict)):
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias", )
        return renamed_pt_tuple_key, pt_tensor
    elif pt_tuple_key[-1] in [
            "weight", "gamma"
    ] and pt_tuple_key[:-1] + ("bias", ) in random_paddle_state_dict:
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias", )
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    if pt_tuple_key[-1] == "weight" and pt_tuple_key[:-1] + (
            "weight", ) in random_paddle_state_dict:
        pt_tuple_key = pt_tuple_key[:-1] + ("weight", )
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight", )
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
        return renamed_pt_tuple_key, pt_tensor

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight", )
    if pt_tuple_key[-1] == "weight":
        pt_tensor = pt_tensor.t()
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight", )
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias", )
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor


def convert_pytorch_state_dict_to_paddle(pt_state_dict, paddle_model):
    # Step 1: Convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    random_paddle_state_dict = paddle_model.state_dict
    paddle_state_dict = {}

    # Need to change some parameters name to match Paddle names
    for pt_key, pt_tensor in pt_state_dict.items():
        renamed_pt_key = rename_key(pt_key)
        pt_tuple_key = tuple(renamed_pt_key.split("."))

        # Correctly rename weight parameters
        paddle_key, paddle_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, pt_tensor, random_paddle_state_dict)

        if paddle_key in random_paddle_state_dict:
            if list(paddle_tensor.shape) != list(
                    random_paddle_state_dict[paddle_key].shape):
                raise ValueError(
                    f"Paddle checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_paddle_state_dict[paddle_key].shape}, but is {paddle_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        paddle_state_dict[paddle_key] = paddle_tensor.numpy()

    return paddle_state_dict
