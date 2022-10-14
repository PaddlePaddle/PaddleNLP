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

DATA_PATH="../dureader-retrieval-baseline-dataset/passage-collection"
TOP_K=50
para_part_cnt=`cat $DATA_PATH/part-00 | wc -l`
python merge.py $para_part_cnt $TOP_K 4 

QUERY2ID="../dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"
PARA2ID="../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"
MODEL_OUTPUT="output/dev.res.top50"
# python metric/convert_recall_res_to_json.py $QUERY2ID $PARA2ID $MODEL_OUTPUT
python ../metric/utils.py --q2id_map $QUERY2ID \
                       --p2id_map $PARA2ID \
                       --recall_result $MODEL_OUTPUT \
                       --outputf output/dual_res.json

REFERENCE_FIEL="../dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/dual_res.json"
python ../metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE