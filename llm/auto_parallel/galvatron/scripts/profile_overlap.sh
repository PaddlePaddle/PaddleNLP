# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5
export NCCL_IB_DISABLE=0
echo "Running python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir output/profile_overlap ./paddlenlp/experimental/galvatron/profiler/profile_overlap.py --output_dir "./output""
python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir output/profile_overlap ./paddlenlp/experimental/galvatron/profiler/profile_overlap.py --output_dir "./output"
sleep 1
rm -r ./profiler_log