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

from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name="paddlenlp_ops",
    ext_modules=[
        CppExtension(
            sources=[
                "./set_stop_value_multi_ends_v2.cc",
                "./set_value_by_flags_and_idx_v2.cc",
                "./get_token_penalty_multi_scores_v2.cc",
                "./get_padding_offset_v2.cc",
                "./update_inputs.cc",
                "./rebuild_padding_v2.cc",
                "../../gpu/save_with_output.cc",
                "../../gpu/save_with_output_msg.cc",
                "../../gpu/get_output.cc",
            ],
            include_dirs=["./plugin/include"],
            extra_objects=["./plugin/build/libxpuplugin.a"],
            extra_compile_args={
                "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-DPADDLE_WITH_XPU"]
            },
        )
    ],
)
