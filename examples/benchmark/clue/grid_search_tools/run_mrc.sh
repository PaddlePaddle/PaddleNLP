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


task_name=$1
MODEL_DIR=$2
cards=$3

if [ ${task_name} == 'chid' ]
then

grd_accu_steps=2
for bs in 6 8
    do
    for lr in 1e-5 2e-5 3e-5
        do
        bash run_chid.sh ${MODEL_DIR} $bs $lr $grd_accu_steps $cards
    done
done
fi


if [ ${task_name} == 'cmrc2018' ]
then 
grd_accu_steps=2
for bs in 16 12
do
    for lr in  1e-5 2e-5 3e-5
    do
        bash run_cmrc.sh ${MODEL_DIR} $bs $lr $grd_accu_steps $cards
    done
done
fi

if [ ${task_name} == 'c3' ]
then

bs=6
grd_accu_steps=3
for lr in 1e-5 2e-5 3e-5
do
    bash run_c3.sh  ${MODEL_DIR} $bs $lr $grd_accu_steps $cards
done

bs=8
grd_accu_steps=4
for lr in 1e-5 2e-5 3e-5
do
    bash run_c3.sh  ${MODEL_DIR} $bs $lr $grd_accu_steps $cards
done

fi
