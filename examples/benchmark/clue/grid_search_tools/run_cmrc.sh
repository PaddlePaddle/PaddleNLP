# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


MODEL_PATH=$1
BATCH_SIZE=$2
LR=$3


git rev-parse HEAD

logdir=${MODEL_PATH}/cmrc2018_log
mkdir -p ${logdir}
python -m paddle.distributed.launch --gpus "$4"  --log_dir ${logdir} ../mrc/run_cmrc.py \
    --model_name_or_path ${MODEL_PATH} \
    --max_seq_length 512 \
    --batch_size ${BATCH_SIZE}  \
    --learning_rate ${LR} \
    --num_train_epochs 2 \
    --logging_steps 100 \
    --output_dir ${MODEL_PATH}/cmrc2018_model/${LR}_${BS}/ \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --do_train \
    --device gpu \
    --gradient_accumulation_steps 2 \
    --save_best_model False \
