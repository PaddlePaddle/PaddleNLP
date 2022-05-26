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

unset GREP_OPTIONS

MODEL_PATH=$1
declare -A dict

for task in afqmc tnews iflytek cmnli ocnli cluewsc2020 csl cmrc2018 chid c3
do
    dict[${task}]=`cat ${MODEL_PATH}/${task}/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
done

echo -e AFQMC"\t"TNEWS"\t"IFLYTEK"\t"CMNLI"\t"OCNLI"\t"CLUEWSC2020"\t"CSL"\t"CMRC2018"\t"CHID"\t"C3

for task in afqmc tnews iflytek cmnli ocnli cluewsc2020 csl cmrc2018 chid c3
do
    echo -e -n "${dict[$task]}\t"
done

echo -e "\n==================================\nbest hyper-paramter list: \n=================================="
for task in afqmc tnews iflytek cmnli ocnli cluewsc2020 csl cmrc2018 chid c3
do
    s=`find  ${MODEL_PATH}/${task}/* | xargs grep -rin "best_acc: ${dict[$task]}"|awk '{split($1, hy, "/"); print(hy[3])}'`
    s=`echo $s|awk '{split($1, hy, "."); print hy[1]"."hy[2]}'`
    s=`echo $s|awk '{split($1, hy, "_"); print hy[1] " " hy[2] " "hy[3]}'`
    echo -e "${task}'s best lr, bs, dropout_p: "$s
done
