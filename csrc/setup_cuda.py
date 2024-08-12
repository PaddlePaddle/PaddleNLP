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

import os

from paddle.utils.cpp_extension import CUDAExtension, setup

library_path = os.environ.get("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")

setup(
    name="paddlenlp_ops",
    ext_modules=CUDAExtension(
        sources=[
            "./generation/save_with_output.cc",
            "./generation/token_penalty_multi_scores_v2.cu",
            "./generation/get_padding_offset_v2.cu",
            "./generation/rebuild_padding_v2.cu",
            "./generation/set_value_by_flags_v2.cu",
            "./generation/stop_generation_multi_ends_v2.cu",
            "./generation/update_inputs.cu",
            "./generation/get_output.cc",
            "./generation/save_with_output_msg.cc",
            "./generation/step.cu",
            "./generation/dequant_int8.cu",
            "./generation/flash_attn_bwd.cc",
            "./generation/tune_cublaslt_gemm.cu",
        ],
        libraries=["cublasLt"],
        library_dirs=[library_path],
    ),
)
