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

cd ..
sed -i -e "s/paddlepaddle/#paddlepaddle/g" requirements-dev.txt
sed -i -e "s/pip install --pre paddlepaddle/#pip install --pre paddlepaddle/g" Makefile

make install

cd ./csrc
pip install -r requirements.txt
wget https://paddle-qa.bj.bcebos.com/benchmark/PaddleNLP/cutlass.tar
tar -xvf cutlass.tar
mv cutlass ./third_party/cutlass
python setup_cuda.py install
