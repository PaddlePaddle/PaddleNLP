# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

unset CUDA_VISIBLE_DEVICES
# gpu
python -u -m paddle.distributed.launch --gpus "0" evaluate.py \
        --device gpu \
        --batch_size 32 \
        --learning_rate 2E-5 \
        --init_from_ckpt "./checkpoints/model_30000/model_state.pdparams" \
        --test_file sort/dev_pairwise.csv

# cpu
# python  evaluate.py \
#         --device cpu \
#         --batch_size 32 \
#         --learning_rate 2E-5 \
#         --init_from_ckpt "./checkpoints/model_30000/model_state.pdparams" \
#         --test_file sort/dev_pairwise.csv