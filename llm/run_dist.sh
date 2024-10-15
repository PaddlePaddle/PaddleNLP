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

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset CUDA_VISIBLE_DEVICES
# LD_LIBRARY_PATH=/opt/software/openmpi-4.0.5/lib 

# 10.3.5.1    g3021
# 10.3.5.2    g3022
# 10.3.6.1    g3023
# 10.3.7.1    g3024
export SAVE_INIT_MODEL=1
#LD_LIBRARY_PATH=/opt/software/openmpi-4.0.5/lib:/home/baidu_test/miniconda3/envs/sci-baidu/lib

#fuser -kv /dev/nvidia*

#conda init && conda activate sci-baidu
python=python

cd ../model_zoo/gpt-3/external_ops/ &&  ${python} setup.py install && cd -

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

PYTHONPATH=../ ${python} -m paddle.distributed.launch \
	--master "10.73.23.7:8678" \
	--nnodes 8 \
	--log_dir log_$(hostname) \
        --gpus 0,1,2,3,4,5,6,7 \
	run_pretrain.py \
        "llama/pretrain-llama_13b-pp4tp2sd2_stage1.json"
