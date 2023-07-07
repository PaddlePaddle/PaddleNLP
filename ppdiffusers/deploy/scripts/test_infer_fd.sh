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
    --device_id 0 \
    --task_name all \
    --infer_op zero_copy_infer \
    --benchmark_steps 10 --parse_prompt_type lpw

python infer.py --model_dir stable-diffusion-v1-5/ \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name all \
    --infer_op raw \
    --benchmark_steps 10 --parse_prompt_type lpw

# test cycle_diffusion
python export_model.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --output_path stable-diffusion-v1-5-cycle_diffusion

python infer.py --model_dir stable-diffusion-v1-5-cycle_diffusion/ \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name cycle_diffusion \
    --infer_op zero_copy_infer \
    --benchmark_steps 10 --parse_prompt_type lpw

python infer.py --model_dir stable-diffusion-v1-5-cycle_diffusion/ \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name cycle_diffusion \
    --infer_op raw \
    --benchmark_steps 10 --parse_prompt_type lpw

# test stable-diffusion-v1-5-inpainting
python export_model.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
    --output_path stable-diffusion-v1-5-inpainting

python infer.py \
    --model_dir stable-diffusion-v1-5-inpainting \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name inpaint \
    --infer_op zero_copy_infer \
    --benchmark_steps 10 --parse_prompt_type lpw

python infer.py \
    --model_dir stable-diffusion-v1-5-inpainting \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name inpaint \
    --infer_op raw \
    --benchmark_steps 10 --parse_prompt_type lpw

# nohup sh test_infer_fd.sh  1> test_infer_fd.log 2>&1 & 

## zero_copy_infer lpw
# ==> Test text2img performance.
# Mean latency: 0.793726 s, p50 latency: 0.793276 s, p90 latency: 0.795765 s, p95 latency: 0.795784 s.
# ==> Test img2img performance.
# Mean latency: 0.662667 s, p50 latency: 0.662543 s, p90 latency: 0.663988 s, p95 latency: 0.664083 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 0.713983 s, p50 latency: 0.713049 s, p90 latency: 0.714997 s, p95 latency: 0.719060 s.
# ==> Test hiresfix performance.
# Mean latency: 1.379235 s, p50 latency: 1.378860 s, p90 latency: 1.381117 s, p95 latency: 1.381354 s.

## raw lpw
# ==> Test text2img performance.
# Mean latency: 0.858216 s, p50 latency: 0.856819 s, p90 latency: 0.863076 s, p95 latency: 0.865202 s.
# ==> Test img2img performance.
# Mean latency: 0.715608 s, p50 latency: 0.713970 s, p90 latency: 0.722402 s, p95 latency: 0.723554 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 0.723306 s, p50 latency: 0.723204 s, p90 latency: 0.725012 s, p95 latency: 0.726074 s.
# ==> Test hiresfix performance.
# Mean latency: 1.449147 s, p50 latency: 1.447849 s, p90 latency: 1.454234 s, p95 latency: 1.454289 s.

## zero_copy_infer lpw ddim no preconfig
# ==> Test cycle diffusion performance.
# Mean latency: 1.347564 s, p50 latency: 1.347331 s, p90 latency: 1.350945 s, p95 latency: 1.352116 s.
## raw lpw
# ==> Test cycle diffusion performance.
# Mean latency: 1.358652 s, p50 latency: 1.356717 s, p90 latency: 1.364285 s, p95 latency: 1.364509 s.

## zero_copy_infer lpw
# ==> Test inpaint performance.
# Mean latency: 0.663649 s, p50 latency: 0.663355 s, p90 latency: 0.664185 s, p95 latency: 0.665098 s.

## raw lpw
# ==> Test inpaint performance.
# Mean latency: 0.716995 s, p50 latency: 0.716940 s, p90 latency: 0.717639 s, p95 latency: 0.717953 s.
