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

# FP16
python infer_dygraph.py --model_dir runwayml/stable-diffusion-v1-5 \
    --task_name all --use_fp16 True \
    --attention_type all --benchmark_steps 5 --device_id 0
python infer_dygraph.py --model_dir runwayml/stable-diffusion-inpainting \
    --task_name inpaint --use_fp16 True \
    --attention_type all --benchmark_steps 5 --device_id 0

# nohup sh test_infer_dygraph.sh  1> test_infer_dygraph.log 2>&1 & 

# # FP32
# python infer_dygraph.py --model_dir runwayml/stable-diffusion-v1-5 \
#     --task_name all --use_fp16 False \
#     --attention_type all --benchmark_steps 5 --device_id 0
# python infer_dygraph.py --model_dir runwayml/stable-diffusion-inpainting \
#     --task_name inpaint --use_fp16 False \
#     --attention_type all --benchmark_steps 5 --device_id 0

# ==> Test text2img performance.
# Mean latency: 1.916538 s, p50 latency: 1.916473 s, p90 latency: 1.917866 s, p95 latency: 1.918224 s.
# ==> Test img2img performance.
# Mean latency: 1.607546 s, p50 latency: 1.607136 s, p90 latency: 1.608465 s, p95 latency: 1.608468 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 2.012481 s, p50 latency: 2.012938 s, p90 latency: 2.013682 s, p95 latency: 2.013842 s.
# ==> Test cycle diffusion performance.
# Mean latency: 2.735736 s, p50 latency: 2.735751 s, p90 latency: 2.735987 s, p95 latency: 2.736011 s.
# ==> Test hiresfix performance.
# Mean latency: 4.009403 s, p50 latency: 4.009006 s, p90 latency: 4.010863 s, p95 latency: 4.011189 s.
# ==> Test text2img performance.
# Mean latency: 1.569671 s, p50 latency: 1.565985 s, p90 latency: 1.577976 s, p95 latency: 1.580268 s.
# ==> Test img2img performance.
# Mean latency: 1.300879 s, p50 latency: 1.307214 s, p90 latency: 1.310239 s, p95 latency: 1.310245 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 1.637774 s, p50 latency: 1.644841 s, p90 latency: 1.646334 s, p95 latency: 1.646553 s.
# ==> Test cycle diffusion performance.
# Mean latency: 2.028485 s, p50 latency: 2.023351 s, p90 latency: 2.037250 s, p95 latency: 2.037850 s.
# ==> Test hiresfix performance.
# Mean latency: 2.424390 s, p50 latency: 2.414912 s, p90 latency: 2.440017 s, p95 latency: 2.440137 s.
# ==> Test text2img performance.
# Mean latency: 1.555480 s, p50 latency: 1.539413 s, p90 latency: 1.580726 s, p95 latency: 1.581641 s.
# ==> Test img2img performance.
# Mean latency: 1.270918 s, p50 latency: 1.263692 s, p90 latency: 1.286539 s, p95 latency: 1.288570 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 1.599304 s, p50 latency: 1.596277 s, p90 latency: 1.606515 s, p95 latency: 1.606969 s.
# ==> Test cycle diffusion performance.
# Mean latency: 1.927718 s, p50 latency: 1.927669 s, p90 latency: 1.928076 s, p95 latency: 1.928148 s.
# ==> Test hiresfix performance.
# Mean latency: 2.226700 s, p50 latency: 2.227113 s, p90 latency: 2.228314 s, p95 latency: 2.228679 s.
# SD1-5 inpainting
# ==> Test inpaint performance.
# Mean latency: 2.001503 s, p50 latency: 2.001337 s, p90 latency: 2.003551 s, p95 latency: 2.004192 s.
# ==> Test inpaint performance.
# Mean latency: 1.553043 s, p50 latency: 1.552598 s, p90 latency: 1.555177 s, p95 latency: 1.555945 s.
# ==> Test inpaint performance.
# Mean latency: 1.596109 s, p50 latency: 1.594528 s, p90 latency: 1.600861 s, p95 latency: 1.601943 s.