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

export TASK_NAME=$1
export LR=$2
export BATCH_SIZE=$3
export PRE_EPOCHS=$4
export SEQ_LEN=$5
export CUDA_VISIBLE_DEVICES=$6
export STUDENT_DIR=$7
export WIDTH_LIST=$8

python -u ./prune.py --model_type ppminilm \
          --model_name_or_path ${STUDENT_DIR} \
          --task_name $TASK_NAME --max_seq_length ${SEQ_LEN}     \
          --batch_size ${BATCH_SIZE}       \
          --learning_rate ${LR}     \
          --num_train_epochs ${PRE_EPOCHS}     \
          --logging_steps 100     \
          --save_steps 100     \
          --output_dir ./pruned_models/$TASK_NAME/0.75/best_model \
          --device gpu  \
          --width_mult_list ${WIDTH_LIST}

