# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


set -eux

unset CUDA_VISIBLE_DEVICES

bs=128
maxlen=128
numH=64
lr=6e-4
maxStep=400000
warmStep=4000
wd=1e-2

teacher=roberta
teacherModel=roberta-wwm-ext-large

alpha=0
beta=1.0
mode=hardest
use_amp=True
teacher_layer_index=19
student_layer_index=5
num_layers=6

hp_config=bs${bs}_maxlen${maxlen}_lr${lr}_wd${wd}_numH${numH}_maxStep${maxStep}_warmStep${warmStep}_adamW_maxnorm1p0_teacher_${teacherModel}_coldboot_teacher_vocab_index${teacher_layer_index}_4l-312d-batchbatch

export PYTHONPATH=../../../../:$PYTHONPATH
output_dir="./pretrain_${hp_config}"

mkdir -p ${output_dir}
cp ./general_distill.py ${output_dir}/
cp ../../../../paddlenlp/transformers/distill_utils.py ${output_dir}/


python3 -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" general_distill.py \
    --model_type ppminilm \
    --num_relation_heads ${numH} \
    --teacher_model_type ${teacher} \
    --teacher_layer_index ${teacher_layer_index} \
    --student_layer_index ${student_layer_index} \
    --teacher_model_name_or_path ${teacherModel} \
    --max_seq_length ${maxlen} \
    --num_layers ${num_layers} \
    --batch_size ${bs} \
    --learning_rate ${lr} \
    --logging_steps 20 \
    --max_steps ${maxStep} \
    --warmup_steps ${warmStep} \
    --save_steps 20000 \
    --weight_decay ${wd} \
    --output_dir ${output_dir} \
    --device gpu \
    --input_dir  dataset/  \
    --use_amp ${use_amp} \
    --alpha ${alpha} \
    --beta ${beta} \
