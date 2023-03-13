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

# python export_model.py --pretrained_model_name_or_path takuma104/control_sd15_canny --output_path control_sd15_canny
export LD_LIBRARY_PATH=./TensorRT-8.5.2.2/lib:$LD_LIBRARY_PATH

# (1)
python fd.py --model_dir control_sd15_canny --scheduler "euler_ancestral" --backend paddle --device gpu --benchmark_steps 10

# (2)
python fd.py --model_dir control_sd15_canny --scheduler "euler_ancestral" --backend paddle_tensorrt --device gpu --benchmark_steps 10 --use_fp16 True

# (3)
python dy.py