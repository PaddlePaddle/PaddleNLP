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

cd ../controlnet
# export LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# test control_sd15_canny
python export_model.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny \
    --output_path control_sd15_canny

python infer.py \
    --model_dir control_sd15_canny \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name all

# nohup sh test_controlnet_infer.sh  1> test_controlnet_infer.log 2>&1 & 