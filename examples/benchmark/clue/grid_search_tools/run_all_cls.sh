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


# For base-size or smaller size model.
# Larger model needs to adjust gradient_accumulation argument or not run all hyperparameters together like this

git rev-parse HEAD

MODEL_DIR=$1
mode=$2

if [ ${mode} == 0 ]
then
id_list="0 1 2 3"
fi

if [ ${mode} == 1 ]
then
id_list="4 5 6 7"
fi

if [ ${mode} == 2 ]
then
id_list="8 9 10 11"
fi

if [ ${mode} == 3 ]
then
id_list="0 1 2 3 4 5 6 7 8 9 10 11"
fi

for hyper_id in ${id_list}
do
    nohup bash launch_one_hyper.sh ${MODEL_DIR} ${hyper_id} &
done
