#!/bin/bash
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.3

data_dir=${1}
conf_path=${2}
ckpt_dir=${3}
predict_data=${4}
learning_rate=${5}
is_train=${6}
max_seq_len=${7}
batch_size=${8}
epoch=${9}
pred_save_path=${10}

python classifier.py --num_epoch ${epoch} \
    --learning_rate 5e-5 \
    --tag_path ${conf_path} \
    --train_data ${data_dir}/train.tsv \
    --dev_data ${data_dir}/dev.tsv \
    --test_data ${data_dir}/test.tsv \
    --predict_data ${predict_data} \
    --do_train ${is_train} \
    --do_predict True \
    --max_seq_len ${max_seq_len} \
    --batch_size ${batch_size} \
    --skip_step 1 \
    --valid_step 1 \
    --checkpoints ${ckpt_dir} \
    --init_ckpt ${ckpt_dir}/best.pdparams \
    --predict_save_path ${pred_save_path} \
    --n_gpu 1 
