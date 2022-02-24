# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

export task=TNEWS

echo Inference of orgin FP32 model
for ((i=0;i<=4;i++));
do
    python infer.py  --task_name ${task} --model_path  ../finetuning/ppminilm-6l-768h/models/${task}/1e-4_64/inference  --use_trt --perf
done

echo After pruning
for ((i=0;i<=4;i++));
do
    python infer.py --task_name ${task} --model_path ../pruning/pruned_models/${task}/0.75/sub_static/float  --use_trt --perf
done

echo After quantization
for ((i=0;i<=4;i++));
do
    python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt --perf
done


