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
TASK_NAME=$2
BATCH_SIZE=$3
LR=$4
GRAD_ACCU_STEPS=$5

export FLAGS_cudnn_deterministic=True

if [ -f "${MODEL_PATH}/${TASK_NAME}/${LR}_${BATCH_SIZE}_0.1.log" ]
then
    # Exits if log exits and best_result is computed.
    best_res=`cat ${MODEL_PATH}/${TASK_NAME}/${LR}_${BATCH_SIZE}_0.1.log |grep "best_result"`
    if [ "${best_res}" != "" ]
    then
        exit 0
    fi
fi
 
if [ $TASK_NAME == 'cmrc2018' ]; then
MAX_SEQ_LEN=512
EPOCHS=2
WARMUP_PROP=0.1
fi

if [ $TASK_NAME == 'c3' ]; then
MAX_SEQ_LEN=512
EPOCHS=8
WARMUP_PROP=0.1
fi

if [ $TASK_NAME == 'chid' ]; then
MAX_SEQ_LEN=64
EPOCHS=3
WARMUP_PROP=0.06
fi

mkdir -p ${MODEL_PATH}/${TASK_NAME}

python ../mrc/run_${TASK_NAME}.py \
    --model_name_or_path ${MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --num_train_epochs ${EPOCHS} \
    --output_dir ${MODEL_PATH}/${TASK_NAME}_model/${LR}_${BATCH_SIZE}/ \
    --do_train \
    --seed 42 \
    --weight_decay 0.01 \
    --device gpu \
    --num_proc 4 \
    --logging_steps 100 \
    --warmup_proportion ${WARMUP_PROP} \
    --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
    --save_best_model False  > ${MODEL_PATH}/${TASK_NAME}/${LR}_${BATCH_SIZE}_0.1.log 2>&1
