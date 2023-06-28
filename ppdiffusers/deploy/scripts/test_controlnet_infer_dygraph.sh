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
    --attention_type all --benchmark_steps 5 --device_id 0

# nohup sh test_controlnet_infer_dygraph.sh  1> test_controlnet_infer_dygraph.log 2>&1 & 

# FP32
# python infer_dygraph.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
#     --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny \
#     --task_name all --use_fp16 False \
#     --attention_type all --benchmark_steps 5 --device_id 0

# ==> Test text2img_control performance.
# Mean latency: 2.722880 s, p50 latency: 2.723120 s, p90 latency: 2.723356 s, p95 latency: 2.723377 s.
# ==> Test img2img_control performance.
# Mean latency: 2.264095 s, p50 latency: 2.263246 s, p90 latency: 2.266077 s, p95 latency: 2.266149 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 2.825745 s, p50 latency: 2.830061 s, p90 latency: 2.831029 s, p95 latency: 2.831048 s.
# ==> Test hiresfix_control performance.
# Mean latency: 5.640611 s, p50 latency: 5.640518 s, p90 latency: 5.642084 s, p95 latency: 5.642484 s.
# ==> Test text2img_control performance.
# Mean latency: 2.368565 s, p50 latency: 2.366539 s, p90 latency: 2.376273 s, p95 latency: 2.378215 s.
# ==> Test img2img_control performance.
# Mean latency: 1.875008 s, p50 latency: 1.869065 s, p90 latency: 1.888904 s, p95 latency: 1.892682 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 2.363184 s, p50 latency: 2.363031 s, p90 latency: 2.366102 s, p95 latency: 2.366777 s.
# ==> Test hiresfix_control performance.
# Mean latency: 3.427918 s, p50 latency: 3.431340 s, p90 latency: 3.434431 s, p95 latency: 3.434833 s.
# ==> Test text2img_control performance.
# Mean latency: 2.293702 s, p50 latency: 2.289995 s, p90 latency: 2.302277 s, p95 latency: 2.306250 s.
# ==> Test img2img_control performance.
# Mean latency: 1.886367 s, p50 latency: 1.886962 s, p90 latency: 1.888798 s, p95 latency: 1.889029 s.
# ==> Test inpaint_legacy_control performance.
# Mean latency: 2.377716 s, p50 latency: 2.377670 s, p90 latency: 2.379309 s, p95 latency: 2.379732 s.
# ==> Test hiresfix_control performance.
# Mean latency: 3.239866 s, p50 latency: 3.239475 s, p90 latency: 3.242614 s, p95 latency: 3.243527 s.