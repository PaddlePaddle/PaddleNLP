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

cd ../slm/model_zoo/gpt/data_tools/
sed -i "s/python3/python/g" Makefile
sed -i "s/python-config/python3.7m-config/g" Makefile
cd -

cd ../slm/model_zoo/gpt-3/data_tools/
sed -i "s/python3/python/g" Makefile
sed -i "s/python-config/python3.7m-config/g" Makefile
cd -

mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy -o .tmp
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz -o .tmp
cd -

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install fast_dataindex -i https://mirror.baidu.com/pypi/simple
python -m pip install setuptools_scm 
python -m pip install Cython 
python -m pip install -r ../requirements.txt  #-i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pybind11 regex sentencepiece tqdm visualdl attrdict pyyaml -i https://mirror.baidu.com/pypi/simple

python -m pip install -e ../
# python -m pip install paddlenlp    # PDC 镜像中安装失败
python -m pip list
