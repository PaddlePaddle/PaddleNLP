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
    --attention_type all --benchmark_steps 5 --device_id 0 --parse_prompt_type lpw
python infer_dygraph.py --model_dir runwayml/stable-diffusion-inpainting \
    --task_name inpaint --use_fp16 True \
    --attention_type all --benchmark_steps 5 --device_id 0 --parse_prompt_type lpw

# nohup sh test_infer_dygraph.sh  1> test_infer_dygraph.log 2>&1 & 

# # FP32
# python infer_dygraph.py --model_dir runwayml/stable-diffusion-v1-5 \
#     --task_name all --use_fp16 False \
#     --attention_type all --benchmark_steps 5 --device_id 0
# python infer_dygraph.py --model_dir runwayml/stable-diffusion-inpainting \
#     --task_name inpaint --use_fp16 False \
#     --attention_type all --benchmark_steps 5 --device_id 0

## raw
# ==> Test text2img performance.
# Mean latency: 1.966795 s, p50 latency: 1.966347 s, p90 latency: 1.967994 s, p95 latency: 1.968173 s.
# ==> Test img2img performance.
# Mean latency: 1.602304 s, p50 latency: 1.601587 s, p90 latency: 1.603798 s, p95 latency: 1.604156 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 2.002676 s, p50 latency: 2.002807 s, p90 latency: 2.003454 s, p95 latency: 2.003653 s.
# ==> Test cycle diffusion performance.
# Mean latency: 2.728374 s, p50 latency: 2.728487 s, p90 latency: 2.729091 s, p95 latency: 2.729288 s.
# ==> Test hiresfix performance.
# Mean latency: 4.037808 s, p50 latency: 4.037807 s, p90 latency: 4.039283 s, p95 latency: 4.039706 s.

## cutlass
# ==> Test text2img performance.
# Mean latency: 1.600378 s, p50 latency: 1.600454 s, p90 latency: 1.602175 s, p95 latency: 1.602312 s.
# ==> Test img2img performance.
# Mean latency: 1.316588 s, p50 latency: 1.315549 s, p90 latency: 1.318886 s, p95 latency: 1.319499 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 1.624272 s, p50 latency: 1.617674 s, p90 latency: 1.638277 s, p95 latency: 1.645022 s.
# ==> Test cycle diffusion performance.
# Mean latency: 2.010290 s, p50 latency: 2.010423 s, p90 latency: 2.010763 s, p95 latency: 2.010806 s.
# ==> Test hiresfix performance.
# Mean latency: 2.413675 s, p50 latency: 2.413513 s, p90 latency: 2.414645 s, p95 latency: 2.414994 s.

## flash
# ==> Test text2img performance.
# Mean latency: 1.552317 s, p50 latency: 1.550014 s, p90 latency: 1.556045 s, p95 latency: 1.556484 s.
# ==> Test img2img performance.
# Mean latency: 1.263727 s, p50 latency: 1.263574 s, p90 latency: 1.266098 s, p95 latency: 1.266871 s.
# ==> Test inpaint_legacy performance.
# Mean latency: 1.592698 s, p50 latency: 1.591540 s, p90 latency: 1.595203 s, p95 latency: 1.595825 s.
# ==> Test cycle diffusion performance.
# Mean latency: 1.914255 s, p50 latency: 1.914288 s, p90 latency: 1.914754 s, p95 latency: 1.914828 s.
# ==> Test hiresfix performance.
# Mean latency: 2.230694 s, p50 latency: 2.230958 s, p90 latency: 2.231327 s, p95 latency: 2.231398 s.

# ==> Test inpaint performance.
# Mean latency: 1.991679 s, p50 latency: 1.991460 s, p90 latency: 1.992477 s, p95 latency: 1.992665 s.
# ==> Test inpaint performance.
# Mean latency: 1.550563 s, p50 latency: 1.550274 s, p90 latency: 1.552293 s, p95 latency: 1.552326 s.
# ==> Test inpaint performance.
# Mean latency: 1.538872 s, p50 latency: 1.539402 s, p90 latency: 1.540718 s, p95 latency: 1.541104 s.
