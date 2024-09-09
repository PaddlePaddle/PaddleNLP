# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

model_name=${model_name:-"facebook/llama-7b"}
output_path=${output_path:-"./llm-inference-output"}
fused_model=${fused_model:-false}
dtype=${dtype:-"float16"}
data_file=${data_file:-"tests/fixtures/llm/zh_query.json"}
decode_strategy=${decode_strategy:-"greedy_search"}
prefix_path=${prefix_path:-"./llm-inference-output"}
export_precache=${export_precache:-"./llm-inference-output"}
top_p=${top_p:-"0.0"}
benchmark=${benchmark:-"0"}


common_arguments="--decode_strategy ${decode_strategy} --src_length 300 --max_length 200 --dtype ${dtype} --batch_size 1 --prefix_path ${prefix_path} --export_precache ${export_precache}"
if [ $fused_model ]; then
    common_arguments+=" --inference_model "
fi
common_arguments+="--data_file ${data_file} --top_p ${top_p} --benchmark ${benchmark} --chat_template none"

cd ..
echo "precache ing"
set -x

echo "==============================run-dynamic-predictor=============================="
python ./llm/predict/predictor.py --model_name_or_path ${model_name} --mode dynamic --output_file ${output_path}/dynamic.json ${common_arguments}

echo "==============================run-export-predictor=============================="
python ./llm/predict/export_model.py --model_name_or_path ${model_name} --output_path ${output_path} ${common_arguments}

echo "==============================run-static-predictor=============================="
python ./llm/predict/predictor.py --model_name_or_path ${output_path} --mode static --output_file ${output_path}/static.json ${common_arguments}


echo "==============================dynamic result=============================="
cat ${output_path}/dynamic.json
echo "==============================static result=============================="
cat ${output_path}/static.json
