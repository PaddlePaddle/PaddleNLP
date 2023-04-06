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

python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 run_eval.py \
--model_path /local_path/glm-130b  \
--eval_path ./wiki.test.tokens \
--batch_size 8 \
--device gpu \
--logging_steps 10 \
--output_dir mp_ckpt \
--overlapping_eval 256  \
--seq_length 1024 

