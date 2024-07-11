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


export TASK_NAME=$1
export LR=$2
export BS=$3
export EPOCH=$4
export MAX_SEQ_LEN=$5
export MODEL_PATH=$6
export grad_acc=$7
export dropout_p=$8

export FLAGS_cudnn_deterministic=True

if [ -f "${MODEL_PATH}/${TASK_NAME}/${LR}_${BS}_${dropout_p}.log" ]
then
    # Exits if log exits and best_result is computed.
    best_acc=`cat ${MODEL_PATH}/${TASK_NAME}/${LR}_${BS}_${dropout_p}.log |grep "best_result"`
    if [ "${best_acc}" != "" ]
    then
        exit 0
    fi
fi

mkdir -p ${MODEL_PATH}/${TASK_NAME}

python -u ../classification/run_clue_classifier.py \
    --model_name_or_path ${MODEL_PATH} \
    --task_name ${TASK_NAME} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --batch_size ${BS}   \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --logging_steps 100 \
    --seed 42  \
    --save_steps  100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ${MODEL_PATH}/models/${TASK_NAME}/${LR}_${BS}/ \
    --device gpu  \
    --gradient_accumulation_steps ${grad_acc} \
    --do_train \
    --dropout ${dropout_p} \
    --save_best_model False > ${MODEL_PATH}/${TASK_NAME}/${LR}_${BS}_${dropout_p}.log 2>&1

