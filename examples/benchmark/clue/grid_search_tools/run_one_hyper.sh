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


lr=$1
bs=$2
card=$3
OUTPUT_DIR=$4

grd=1

if [ $bs == 64 ]
then
grd=2
fi


sh run_clue_classifier.sh AFQMC $lr $bs 3 128 $card ${OUTPUT_DIR} 1 0.1 > ${OUTPUT_DIR}/afqmc/${lr}_${bs}_3_128.log


sh run_clue_classifier.sh TNEWS $lr $bs 3 128 $card ${OUTPUT_DIR} 1 0.1 > ${OUTPUT_DIR}/tnews/${lr}_${bs}_3_128.log

sh run_clue_classifier.sh IFLYTEK $lr $bs 3 128 $card ${OUTPUT_DIR} 1 0.1 > ${OUTPUT_DIR}/ifly/${lr}_${bs}_3_0.1_128.log
sh run_clue_classifier.sh IFLYTEK $lr $bs 3 128 $card ${OUTPUT_DIR} 1 0.0 > ${OUTPUT_DIR}/ifly/${lr}_${bs}_3_0.0_128.log

sh run_clue_classifier.sh OCNLI $lr $bs 5 128 $card ${OUTPUT_DIR} 1 0.1 > ${OUTPUT_DIR}/ocnli/${lr}_${bs}_5_128.log

sh run_clue_classifier.sh CLUEWSC2020 $lr $bs 50 128 $card ${OUTPUT_DIR} 1 0.0  > ${OUTPUT_DIR}/wsc/${lr}_${bs}_50_0.0_128.log
sh run_clue_classifier.sh CLUEWSC2020 $lr $bs 50 128 $card ${OUTPUT_DIR} 1 0.1  > ${OUTPUT_DIR}/wsc/${lr}_${bs}_50_0.1_128.log
sh run_clue_classifier.sh CLUEWSC2020 $lr 8 50 128 $card ${OUTPUT_DIR} 1 0.0 > ${OUTPUT_DIR}/wsc/${lr}_8_50_0.0_128.log
sh run_clue_classifier.sh CLUEWSC2020 $lr 8 50 128 $card ${OUTPUT_DIR} 1 0.1 > ${OUTPUT_DIR}/wsc/${lr}_8_50_0.1_128.log

sh run_clue_classifier.sh CSL $lr $bs 5 256 $card ${OUTPUT_DIR} ${grd} 0.1 > ${OUTPUT_DIR}/csl/${lr}_${bs}_5_128.log

sh run_clue_classifier.sh CMNLI $lr $bs 2 128 $card ${OUTPUT_DIR} 1 0.1 > ${OUTPUT_DIR}/cmnli/${lr}_${bs}_2_128.log
