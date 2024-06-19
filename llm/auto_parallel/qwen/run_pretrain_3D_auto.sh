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

# just for debug

set -x
unset CUDA_VISIBLE_DEVICES

export FLAGS_call_stack_level=3
export FLAGS_use_cuda_managed_memory=true

task_name="llama_auto_dp2mp2pp2"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH


rm -rf ./auto_3d

export FLAGS_embedding_deterministic=1        
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_enable_pir_in_executor=0

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "auto_3d" \
    run_pretrain_3D_auto.py ./pretrain_argument_auto_dp2tp2pp2.json
