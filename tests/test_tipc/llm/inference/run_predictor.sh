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

export FLAGS_control_flow_use_new_executor=1
export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.92

model_name=${model_name:-"facebook/llama-7b"}
output_path=${output_path:-"./llm-inference-output"}
fused_model=${fused_model:-false}
dtype=${dtype:-"float16"}
inference_model=${inference_model:-"true"}
decode_strategy=${decode_strategy:-"greedy_search"}
top_p=${top_p:-"0.0"}
data_file=${data_file:-"tests/fixtures/llm/zh_query.json"}
benchmark=${benchmark:-"0"}

common_arguments="--decode_strategy ${decode_strategy} --src_length 300 --max_length 200 --benchmark ${benchmark} --dtype ${dtype} --batch_size 3 --inference_model ${inference_model} "
common_arguments+="--data_file ${data_file} --top_p ${top_p} --chat_template none"

echo "pwd -> "

cd ..

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
