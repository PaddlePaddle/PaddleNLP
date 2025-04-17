#!/usr/bin/env python3

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved.

Build and setup XPU custom ops for ERNIE Bot.
"""

import os

import paddle
from paddle.utils.cpp_extension import CppExtension, setup

PADDLE_PATH = os.path.dirname(paddle.__file__)
PADDLE_INCLUDE_PATH = os.path.join(PADDLE_PATH, "include")
PADDLE_LIB_PATH = os.path.join(PADDLE_PATH, "libs")

XPU_LIB = os.getenv("XPU_LIB")

XFT_PATH = f"{XPU_LIB}/xft_output"
if XFT_PATH is None:
    XFT_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xft")
    XFT_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libxft.so")
else:
    XFT_INC_PATH = os.path.join(XFT_PATH, "include")
    XFT_LIB_PATH = os.path.join(XFT_PATH, "so", "libxft.so")


XRE_PATH = f"{XPU_LIB}/xre"
XRE_INC_PATH = os.path.join(XRE_PATH, "include")
XRE_LIB_PATH = os.path.join(XRE_PATH, "so", "libcudart.so")

XFA_PATH = f"{XPU_LIB}/xhpc/xfa"
XFA_INC_PATH = os.path.join(XFA_PATH, "include")
XFA_LIB_PATH = os.path.join(XFA_PATH, "so", "libxpu_flash_attention.so")

XBLAS_PATH = f"{XPU_LIB}/xhpc/xblas"
XBLAS_INC_PATH = os.path.join(XBLAS_PATH, "include")
XBLAS_LIB_PATH = os.path.join(XBLAS_PATH, "so", "libxpu_blas.so")

setup(
    name="paddlenlp_ops",
    ext_modules=[
        CppExtension(
            sources=[
                "./update_inputs_v2.cc",
                "./set_preids_token_penalty_multi_scores.cc",
                "./set_stop_value_multi_ends_v2.cc",
                "./set_value_by_flags_and_idx_v2.cc",
                "./get_token_penalty_multi_scores_v2.cc",
                "./get_padding_offset_v2.cc",
                # "./update_inputs.cc",
                "./rebuild_padding_v2.cc",
                "../../gpu/save_with_output.cc",
                "../../gpu/save_with_output_msg.cc",
                "../../gpu/get_output.cc",
                "./moe_dispatch.cc",
                "./moe_ffn.cc",
                "./moe_reduce.cc",
                "./mla_block_multihead_attention_xpu.cc",
                "./weight_only_linear.cc",
                "./get_position_ids.cc",
                "./get_position_ids_v2.cc",
                "./adjust_batch.cc",
                "./gather_next_token.cc",
                "step.cc",
            ],
            include_dirs=[".", "./plugin/include", XRE_INC_PATH, XFT_INC_PATH, XFA_INC_PATH, XBLAS_INC_PATH],
            extra_objects=["./plugin/build/libxpuplugin.a", XRE_LIB_PATH, XFT_LIB_PATH, XFA_LIB_PATH, XBLAS_LIB_PATH],
            extra_compile_args={"cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-DPADDLE_WITH_XPU"]},
        )
    ],
)
