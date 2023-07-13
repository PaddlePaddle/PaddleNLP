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
pip install diffusers==0.17.1

python infer_dygraph_toch.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny \
    --task_name all --use_fp16 True \
    --attention_type raw --benchmark_steps 10 --device_id 0 --parse_prompt_type raw

python infer_dygraph_toch.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny \
    --task_name all --use_fp16 True \
    --attention_type sdp --benchmark_steps 10 --device_id 0 --parse_prompt_type raw

python infer_dygraph_toch.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny \
    --task_name all --use_fp16 True \
    --attention_type sdp --benchmark_steps 10 --device_id 0 --parse_prompt_type raw --compile True --channels_last True

# nohup sh test_controlnet_infer_dygraph_torch.sh  1> test_controlnet_infer_dygraph_torch.log 2>&1 & 
