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

# install specific version of paddlepaddle
pip uninstall -y paddlepaddle-gpu
wget https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/9bf5a86f13aa85b38715c015693f973290c8f9da/paddlepaddle_gpu-0.0.0.post120-cp310-cp310-linux_x86_64.whl
pip install --force-reinstall paddlepaddle_gpu-0.0.0.post120-cp310-cp310-linux_x86_64.whl

# install requirements
cd ..
sed -i -e "s/paddlepaddle/#paddlepaddle/g" requirements-dev.txt

make install

# install csrc package
cd ./csrc
pip install pybind11 cupy-cuda12x
python setup_cuda.py install
