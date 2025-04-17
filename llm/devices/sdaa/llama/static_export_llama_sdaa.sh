#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

export FLAGS_enable_pir_api=0
SDAA_VISIBLE_DEVICES=0,1,2,3 PADDLE_XCCL_BACKEND=sdaa \
python -m paddle.distributed.launch  ./../../../predict/export_model.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --output_path ./output_dir/exported_model/llama2_13b_chat_wint8_block_size32 --dtype float16 --block_attn  --quant_type weight_only_int8 --device sdaa --block_size 32

