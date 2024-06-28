# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

cd ../legacy/model_zoo/moe/data_tools/
sed -i "s/python3/python3.10/g" Makefile
cd -

python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
unset http_proxy https_proxy
python3 -m pip install -r ../requirements.txt #-i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install pybind11 regex sentencepiece tqdm visualdl #-i https://mirror.baidu.com/pypi/simple
python3 -m pip install --upgrade paddlenlp
# get data
cd ../legacy/model_zoo/moe/dygraph/
rm -rf data
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt2/train.data.json_ids.npz