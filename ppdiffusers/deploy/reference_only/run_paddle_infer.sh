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

# develop ppdiffusers paddlepaddle  1.0.7的fastdeploy
python export.py --pretrained_model_name_or_path TASUKU2023/Chilloutmix --output_path Chilloutmix
python infer.py --model_dir Chilloutmix --backend paddle --parse_prompt_type raw --infer_op raw