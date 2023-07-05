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
    --task_name all \
    --infer_op zero_copy_infer \
    --benchmark_steps 10 --parse_prompt_type lpw

python infer.py \
    --model_dir control_sd15_canny \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name all \
    --infer_op raw \
    --benchmark_steps 10 --parse_prompt_type lpw
# nohup sh test_controlnet_infer_fd.sh  1> test_controlnet_infer_fd.log 2>&1 & 

## zero_copy_infer lpw
# ==> Test text2img_control performance.
# Mean latency: 1.116596 s, p50 latency: 1.116411 s, p90 latency: 1.117224 s, p95 latency: 1.117570 s.
# ==> Test img2img_control performance.
# Mean latency: 0.923659 s, p50 latency: 0.923360 s, p90 latency: 0.924491 s, p95 latency: 0.924889 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 0.990658 s, p50 latency: 0.990560 s, p90 latency: 0.991377 s, p95 latency: 0.991382 s.
# ==> Test hiresfix_control performance.
# Mean latency: 1.941339 s, p50 latency: 1.940671 s, p90 latency: 1.943967 s, p95 latency: 1.944597 s.

## raw lpw
# ==> Test text2img_control performance.
# Mean latency: 1.316932 s, p50 latency: 1.316731 s, p90 latency: 1.318196 s, p95 latency: 1.318324 s.
# ==> Test img2img_control performance.
# Mean latency: 1.081752 s, p50 latency: 1.081734 s, p90 latency: 1.081981 s, p95 latency: 1.082050 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 1.088706 s, p50 latency: 1.088135 s, p90 latency: 1.090777 s, p95 latency: 1.091583 s.
# ==> Test hiresfix_control performance.
# Mean latency: 2.213738 s, p50 latency: 2.212821 s, p90 latency: 2.214525 s, p95 latency: 2.218274 s.
