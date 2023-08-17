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

export PYTHONPATH=$PYTHONPATH:../../../

python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
            export_generation_model.py \
            --model_name_or_dir=/root/paddlejob/workspace/output/zhengshifeng/Chatglm/FastLLMDeploy/models/chatglm/THUDM/chatglm-6b  \
            --model_dtype=float16 \
            --output_dir=./inference_model_float_16_rank8 2>&1
