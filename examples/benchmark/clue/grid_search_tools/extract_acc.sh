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

afqmc=`cat ${MODEL_PATH}/afqmc/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
tnews=`cat ${MODEL_PATH}/tnews/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
ifly=`cat ${MODEL_PATH}/ifly/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
cmnli=`cat ${MODEL_PATH}/cmnli/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
ocnli=`cat ${MODEL_PATH}/ocnli/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
wsc=`cat ${MODEL_PATH}/wsc/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
csl=`cat ${MODEL_PATH}/csl/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`

cmrc2018=`cat ${MODEL_PATH}/cmrc2018_log/workerlog.0|grep best_res|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
chid=`cat ${MODEL_PATH}/chid_log/workerlog.0|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
c3=`cat  ${MODEL_PATH}/c3_log/workerlog.0|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`

echo AFQMC"\t"TNEWS"\t"IFLYTEK"\t"CMNLI"\t"OCNLI"\t"CLUEWSC2020"\t"CSL"\t"CMRC2018"\t"CHID"\t"C3
echo  ${afqmc}"\t"${tnews}"\t"${ifly}"\t"${cmnli}"\t"${ocnli}"\t"${wsc}"\t"${csl}"\t"${cmrc2018}"\t"${chid}"\t"${c3}

