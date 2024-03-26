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


param="model_name_or_path=qwen "
param="model_item=CE_qwen-7b-chat "
param+="run_mode=MP1-mbs2 "
param+="batch_size=2 "
param+="device_num=N1C1 "
param+="dtype=bf16 "

bash ./test_tipc/llm/prepare.sh

bash -c "${param} bash ./test_tipc/llm/run_ce.sh"

