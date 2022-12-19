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

export CUDA_VISIBLE_DEVICES=0

model_type=$1

if [ ! $model_type ]; then
echo "Please enter the correct export model type, for example: sh run_export extraction"
elif [ $model_type = extraction ]; then 
python  export_model.py \
        --model_type "extraction" \
        --model_path "./checkpoints/ext_checkpoints/best.pdparams" \
        --save_path "./checkpoints/ext_checkpoints/static/infer" 

elif [ $model_type = classification ]; then
python  export_model.py \
        --model_type "classification" \
        --model_path "./checkpoints/cls_checkpoints/best.pdparams" \
        --save_path "./checkpoints/cls_checkpoints/static/infer" 
        
elif [ $model_type = pp_minilm ]; then
python  export_model.py \
        --model_type "pp_minilm" \
        --base_model_name "ppminilm-6l-768h" \
        --model_path "./checkpoints/pp_checkpoints/best.pdparams" \
        --save_path "./checkpoints/pp_checkpoints/static/infer" 
else
echo "Three model_types are supported:  [extraction, classification, pp_minilm]"
fi
