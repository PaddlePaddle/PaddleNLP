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

cd ../slm/model_zoo/gpt-3/data_tools/
sed -i "s/python3/python3.7/g" Makefile
cd -

python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r ../requirements.txt #-i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pybind11 regex sentencepiece tqdm visualdl -i https://mirror.baidu.com/pypi/simple
python -m pip install fast_dataindex -i https://mirror.baidu.com/pypi/simple

# get data
cd ../slm/model_zoo/gpt-3/static/
mkdir train_data && cd train_data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt2/train.data.json_ids.npz
