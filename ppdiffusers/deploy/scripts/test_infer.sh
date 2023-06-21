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

cd ..
# export LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# test stable-diffusion-v1-5
python export_model.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --output_path stable-diffusion-v1-5

python infer.py --model_dir stable-diffusion-v1-5/ \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --task_name all

# test cycle_diffusion
python export_model.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --output_path stable-diffusion-v1-5-cycle_diffusion

python infer.py --model_dir stable-diffusion-v1-5-cycle_diffusion/ \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --task_name cycle_diffusion

# test stable-diffusion-v1-5-inpainting
python export_model.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
    --output_path stable-diffusion-v1-5-inpainting

python infer.py \
    --model_dir stable-diffusion-v1-5-inpainting \
    --scheduler euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name inpaint

# nohup sh test_infer.sh  1> test_infer.log 2>&1 & 