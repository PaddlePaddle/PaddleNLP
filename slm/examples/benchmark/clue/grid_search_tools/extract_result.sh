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
    # `awk '{print substr($0,1,11)}'` is to prevent '[' brought by logger.
    if [ $task == 'cmrc2018' ]; then
        dict[${task}]=`cat ${MODEL_PATH}/${task}/*|grep best_result|awk '{print $7}' |awk '{print substr($0,1,11)}'|awk 'BEGIN {max = 0} {if ($1+0 > max+0) max=$1} END {print  max}'`
    else
    dict[${task}]=`tail -n 1  ${MODEL_PATH}/${task}/*|grep best_result|awk '{print $7}'|awk '{print substr($0,1,5)}'|awk 'BEGIN {max = 0} {if ($1+0 > max+0) max=$1} END {print  max}'`
    fi
done

echo -e AFQMC"\t"TNEWS"\t"IFLYTEK"\t"CMNLI"\t"OCNLI"\t"CLUEWSC2020"\t"CSL"\t"CMRC2018"\t"CHID"\t"C3

for task in afqmc tnews iflytek cmnli ocnli cluewsc2020 csl cmrc2018 chid c3
do
    echo -e -n "${dict[$task]}\t"
done

echo -e "\n====================================================================\nBest hyper-parameters list: \n===================================================================="
echo -e TASK"\t"result"\t(lr, batch_size, dropout_p)"

for task in afqmc tnews iflytek cmnli ocnli cluewsc2020 csl cmrc2018 chid c3
do
    if [ -z ${dict[$task]} ]
    then
    continue
    fi
    s=`find  ${MODEL_PATH}/${task}/* | xargs grep -rin "best_result: ${dict[$task]}"`
    if [ $task == 'cmrc2018' ]; then
    s=${s%/*}
    fi
    s=${s##*/}
    s=`echo $s|awk '{split($1, hy, "."); print hy[1]"."hy[2]}'`
    s=`echo $s|awk '{split($1, hy, "_"); print hy[1]"," hy[2]"," hy[3]}'`
    echo -n ${task}| tr 'a-z' 'A-Z'
    echo -e "\t"${dict[$task]}"\t("$s")"
done
