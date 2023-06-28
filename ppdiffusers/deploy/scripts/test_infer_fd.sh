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
    --benchmark_steps 10

python infer.py --model_dir stable-diffusion-v1-5/ \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name all \
    --infer_op raw \
    --benchmark_steps 10

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
    --benchmark_steps 10

python infer.py --model_dir stable-diffusion-v1-5-cycle_diffusion/ \
    --scheduler preconfig-euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name cycle_diffusion \
    --infer_op raw \
    --benchmark_steps 10

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
    --task_name inpaint \
    --infer_op zero_copy_infer \
    --benchmark_steps 10

python infer.py \
    --model_dir stable-diffusion-v1-5-inpainting \
    --scheduler euler-ancestral \
    --backend paddle_tensorrt \
    --use_fp16 True \
    --device gpu \
    --device_id 0 \
    --task_name inpaint \
    --infer_op raw \
    --benchmark_steps 10

# nohup sh test_infer_fd.sh  1> test_infer_fd.log 2>&1 & 

# zero_copy_infer
# ==> Test text2img performance.
# Mean latency: 0.791007 s, p50 latency: 0.789686 s, p90 latency: 0.791581 s, p95 latency: 0.796705 s.
# ==> Test img2img performance.
# Mean latency: 0.703892 s, p50 latency: 0.703706 s, p90 latency: 0.704755 s, p95 latency: 0.704946 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 0.715938 s, p50 latency: 0.715886 s, p90 latency: 0.716202 s, p95 latency: 0.716303 s.
# ==> Test hiresfix performance.
# Mean latency: 1.438618 s, p50 latency: 1.438173 s, p90 latency: 1.439861 s, p95 latency: 1.440739 s.

# raw
# ==> Test text2img performance.
# Mean latency: 0.864235 s, p50 latency: 0.864001 s, p90 latency: 0.864603 s, p95 latency: 0.865531 s.
# ==> Test img2img performance.
# Mean latency: 0.715326 s, p50 latency: 0.714759 s, p90 latency: 0.715858 s, p95 latency: 0.718002 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 0.726799 s, p50 latency: 0.726491 s, p90 latency: 0.727381 s, p95 latency: 0.728085 s.
# ==> Test hiresfix performance.
# Mean latency: 1.452914 s, p50 latency: 1.453080 s, p90 latency: 1.453319 s, p95 latency: 1.453735 s.

# zero_copy_infer
# ==> Test cycle diffusion performance.
# Mean latency: 1.320150 s, p50 latency: 1.319936 s, p90 latency: 1.321060 s, p95 latency: 1.321400 s.

# raw
# ==> Test cycle diffusion performance.
# Mean latency: 1.352944 s, p50 latency: 1.351467 s, p90 latency: 1.355291 s, p95 latency: 1.359292 s.

# zero_copy_infer
# ==> Test inpaint performance.
# Mean latency: 0.714265 s, p50 latency: 0.714164 s, p90 latency: 0.715602 s, p95 latency: 0.716116 s.
# raw
# ==> Test inpaint performance.
# Mean latency: 0.727461 s, p50 latency: 0.727120 s, p90 latency: 0.728364 s, p95 latency: 0.728885 s.