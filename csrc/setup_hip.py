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
import subprocess

def update_git_submodule():
    try:
        subprocess.run(["git", "submodule", "update", "--init"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while updating git submodule: {str(e)}")
        raise
update_git_submodule()
setup(
    name="paddlenlp_ops",
    ext_modules=CUDAExtension(
        sources=[
            "./gpu/save_with_output.cc",
            "./gpu/set_value_by_flags.cu",
            "./gpu/token_penalty_multi_scores.cu",
            "./gpu/token_penalty_multi_scores_v2.cu",
            "./gpu/stop_generation_multi_ends.cu",
            "./gpu/fused_get_rope.cu",
            "./gpu/get_padding_offset.cu",
            "./gpu/qkv_transpose_split.cu",
            "./gpu/rebuild_padding.cu",
            "./gpu/transpose_removing_padding.cu",
            "./gpu/write_cache_kv.cu",
            "./gpu/encode_rotary_qk.cu",
            "./gpu/get_padding_offset_v2.cu",
            "./gpu/rebuild_padding_v2.cu",
            "./gpu/set_value_by_flags_v2.cu",
            "./gpu/stop_generation_multi_ends_v2.cu",
            "./gpu/update_inputs.cu",
            "./gpu/get_output.cc",
            "./gpu/save_with_output_msg.cc",
            "./gpu/write_int8_cache_kv.cu",
            "./gpu/step.cu",
            "./gpu/quant_int8.cu",
            "./gpu/dequant_int8.cu",
            "./gpu/flash_attn_bwd.cc",
            "./gpu/update_inputs_v2.cu",
            "./gpu/set_preids_token_penalty_multi_scores.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "hipcc": [
                "-O3",
                "--gpu-max-threads-per-block=1024",
                "-U__HIP_NO_HALF_OPERATORS__",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_BFLOAT16_OPERATORS__",
                "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
                "-U__HIP_NO_BFLOAT162_OPERATORS__",
                "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
                "-Ithird_party/cutlass/include",
                "-Ithird_party/nlohmann_json/single_include",
            ],
        },
    ),
)
