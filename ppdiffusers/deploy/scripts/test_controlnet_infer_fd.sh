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
    --benchmark_steps 10

python infer.py \
    --model_dir control_sd15_canny \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name all \
    --infer_op raw \
    --benchmark_steps 10
# nohup sh test_controlnet_infer_fd.sh  1> test_controlnet_infer_fd.log 2>&1 & 

# ==> Test text2img_control performance.
# Mean latency: 1.124626 s, p50 latency: 1.123942 s, p90 latency: 1.127096 s, p95 latency: 1.128142 s.
# ==> Test img2img_control performance.
# Mean latency: 1.006784 s, p50 latency: 1.007280 s, p90 latency: 1.008598 s, p95 latency: 1.008995 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 1.010847 s, p50 latency: 1.010749 s, p90 latency: 1.013414 s, p95 latency: 1.013958 s.
# ==> Test hiresfix_control performance.
# Mean latency: 2.047749 s, p50 latency: 2.047972 s, p90 latency: 2.049623 s, p95 latency: 2.051662 s.

# ==> Test text2img_control performance.
# Mean latency: 1.317447 s, p50 latency: 1.317423 s, p90 latency: 1.319283 s, p95 latency: 1.319428 s.
# ==> Test img2img_control performance.
# Mean latency: 1.092353 s, p50 latency: 1.092538 s, p90 latency: 1.094368 s, p95 latency: 1.094635 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 1.118845 s, p50 latency: 1.102380 s, p90 latency: 1.155136 s, p95 latency: 1.155555 s.
# ==> Test hiresfix_control performance.
# Mean latency: 2.326568 s, p50 latency: 2.326191 s, p90 latency: 2.328305 s, p95 latency: 2.329674 s.