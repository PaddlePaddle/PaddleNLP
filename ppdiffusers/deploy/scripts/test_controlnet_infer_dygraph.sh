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

# FP16
python infer_dygraph.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny \
    --task_name all --use_fp16 True \
    --attention_type all --benchmark_steps 5 --device_id 0 --parse_prompt_type lpw

# nohup sh test_controlnet_infer_dygraph.sh  1> test_controlnet_infer_dygraph.log 2>&1 & 

# FP32
# python infer_dygraph.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
#     --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny \
#     --task_name all --use_fp16 False \
#     --attention_type all --benchmark_steps 5 --device_id 0

# # raw
# ==> Test text2img_control performance.
# Mean latency: 2.765781 s, p50 latency: 2.765087 s, p90 latency: 2.767249 s, p95 latency: 2.767289 s.
# ==> Test img2img_control performance.
# Mean latency: 2.249113 s, p50 latency: 2.247525 s, p90 latency: 2.252412 s, p95 latency: 2.253428 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 2.798517 s, p50 latency: 2.798506 s, p90 latency: 2.799392 s, p95 latency: 2.799600 s.
# ==> Test hiresfix_control performance.
# Mean latency: 5.625017 s, p50 latency: 5.624560 s, p90 latency: 5.626832 s, p95 latency: 5.627412 s.

# # cutlass
# ==> Test text2img_control performance.
# Mean latency: 2.221491 s, p50 latency: 2.219046 s, p90 latency: 2.226926 s, p95 latency: 2.227513 s.
# ==> Test img2img_control performance.
# Mean latency: 1.845735 s, p50 latency: 1.845492 s, p90 latency: 1.847032 s, p95 latency: 1.847059 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 2.299109 s, p50 latency: 2.299197 s, p90 latency: 2.300616 s, p95 latency: 2.300845 s.
# ==> Test hiresfix_control performance.
# Mean latency: 3.397126 s, p50 latency: 3.397122 s, p90 latency: 3.398814 s, p95 latency: 3.399294 s.

# # flash
# ==> Test text2img_control performance.
# Mean latency: 2.247151 s, p50 latency: 2.245066 s, p90 latency: 2.251284 s, p95 latency: 2.252165 s.
# ==> Test img2img_control performance.
# Mean latency: 1.831867 s, p50 latency: 1.832337 s, p90 latency: 1.832757 s, p95 latency: 1.832806 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 2.291782 s, p50 latency: 2.290435 s, p90 latency: 2.298838 s, p95 latency: 2.300244 s.
# ==> Test hiresfix_control performance.
# Mean latency: 3.166110 s, p50 latency: 3.164805 s, p90 latency: 3.171124 s, p95 latency: 3.172821 s.
