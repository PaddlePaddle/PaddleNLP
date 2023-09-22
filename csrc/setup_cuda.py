# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name="paddlenlp_ops",
    ext_modules=CUDAExtension(
        sources=[
            "./generation/save_with_output.cc",
            "./generation/set_mask_value.cu",
            "./generation/set_value_by_flags.cu",
            "./generation/token_penalty_multi_scores.cu",
            "./generation/stop_generation_multi_ends.cu",
            "./generation/fused_get_rope.cu",
            "./generation/get_padding_offset.cu",
            "./generation/qkv_transpose_split.cu",
            "./generation/rebuild_padding.cu",
            "./generation/transpose_removing_padding.cu",
            "./generation/write_cache_kv.cu",
            "./generation/encode_rotary_qk.cu",
            "./generation/top_p_sampling.cu",
            "./generation/set_alibi_mask_value.cu",
        ]
    ),
)
