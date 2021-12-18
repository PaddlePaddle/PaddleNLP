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

task = tnews
echo Inference of orgin FP32 model
python  infer.py  --task_name ${task} --model_path  tnews/float  --use_trt  --collect_shape --perf
python  infer.py  --task_name ${task} --model_path  tnews/float  --use_trt --perf
python  infer.py  --task_name ${task} --model_path  tnews/float  --use_trt --perf
python  infer.py  --task_name ${task} --model_path  tnews/float  --use_trt --perf
python  infer.py  --task_name ${task} --model_path  tnews/float  --use_trt --perf
python  infer.py  --task_name ${task} --model_path  tnews/float  --use_trt --perf


echo After pruning
python infer.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt --collect_shape --perf
python infer.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt --perf
python infer.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt --perf
python infer.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt --perf
python infer.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt --perf
python infer.py --task_name ${task} --model_path ofa_models/TNEWS/0.75/sub_static/float  --use_trt --perf

echo After quantization
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt  --collect_shape --perf
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt --perf
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt --perf
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt --perf
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt --perf
python  infer.py  --task_name tnews --model_path  ../quantization/${task}_quant_models/mse4/int8  --int8 --use_trt --perf


