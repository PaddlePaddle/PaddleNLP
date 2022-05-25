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


# $1: Model name or directory
# $2: Batch_size
# $3: Learning rate
# $4: Gradient accumulation steps
# $5: Card id(s)

MODEL_PATH=$1
BATCH_SIZE=$2
LR=$3
GRAD_ACCU_STEPS=$4
CARD_ID=$5


logdir=${MODEL_PATH}/c3_log
mkdir -p ${logdir}
python -m paddle.distributed.launch --gpus "$CARD_ID" --log_dir ${logdir} ../mrc/run_c3.py \
    --model_name_or_path ${MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --max_seq_length 512 \
    --num_train_epochs 8 \
    --output_dir ${MODEL_PATH}/c3_model/${LR}_${BATCH_SIZE}/ \
    --do_train \
    --warmup_proportion 0.1 \
    --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
    --save_best_model False \
