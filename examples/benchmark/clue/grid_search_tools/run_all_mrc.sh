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


git rev-parse HEAD

MODEL_DIR=$1
mode=$2

# mode 0: 4 cards for CHID
if [ ${mode} == 0 ]
then
task_name=chid
cards="0,1,2,3"
nohup bash run_mrc.sh ${task_name} ${MODEL_DIR} ${cards} &
fi

# mode 1: 4 cards for C3
if [ ${mode} == 1 ]
then
task_name=c3
cards="0,1,2,3"
nohup bash run_mrc.sh ${task_name} ${MODEL_DIR} ${cards} &
fi

# mode 2: 2 cards for CMRC2018
if [ ${mode} == 2 ]
then
task_name=cmrc2018
cards="0,1"
nohup bash run_mrc.sh ${task_name} ${MODEL_DIR} ${cards} &
fi


# mode 3: 8 cards for CHID, C3 and CMRC2018
if [ ${mode} == 3 ]
then
task_name=chid
cards="0,1,2,3"
nohup bash run_mrc.sh ${task_name} ${MODEL_DIR} ${cards} &

task_name=c3
cards="4,5,6,7"
nohup bash run_mrc.sh ${task_name} ${MODEL_DIR} ${cards} &

task_name=cmrc2018
cards="0,1"
nohup bash run_mrc.sh ${task_name} ${MODEL_DIR} ${cards} &
fi
