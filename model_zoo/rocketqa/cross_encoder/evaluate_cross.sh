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

MODEL_OUTPUT="output/result.txt"
ID_MAP="../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv"
python ../metric/utils.py --score_f $MODEL_OUTPUT \
                       --id_f $ID_MAP \
                       --mode rank \
                       --outputf output/cross_res.json
REFERENCE_FIEL="../dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/cross_res.json"
python ../metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE