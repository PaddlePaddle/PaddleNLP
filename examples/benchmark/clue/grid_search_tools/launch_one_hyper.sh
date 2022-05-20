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

OUTPUT_DIR=$1
card=$2

for task in afqmc tnews ifly ocnli wsc csl cmnli
do
    mkdir -p ${OUTPUT_DIR}/${task}
done


if [ "$card" == "0" ]
then
lr=1e-5
bs=16
fi

if [ $card == "1" ]
then
lr=2e-5
bs=16
fi

if [ $card == "2" ]
then
lr=3e-5
bs=16
fi

if [ $card == "3" ]
then
lr=5e-5
bs=16
fi

if [ $card == "4" ]
then
lr=1e-5
bs=32
fi

if [ $card == "5" ]
then
lr=2e-5
bs=32
fi

if [ $card == "6" ]
then
lr=3e-5
bs=32
fi

if [ $card == "7" ]
then
lr=5e-5
bs=32
fi


if [ $card == "8" ]
then
lr=1e-5
bs=64
card=5
fi

if [ $card == "9" ]
then
lr=2e-5
bs=64
card=3
fi

if [ $card == "10" ]
then
lr=3e-5
bs=64
card=6
fi

if [ $card == "11" ]
then
lr=5e-5
bs=64
card=7
fi

bash run_one_hyper.sh $lr $bs  $card $OUTPUT_DIR
