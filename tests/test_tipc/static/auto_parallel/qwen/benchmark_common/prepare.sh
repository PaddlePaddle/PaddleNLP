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

python -m pip install -r ../requirements.txt
python -m pip install -r ../requirements-dev.txt

# install fused_ln custom ops
cd ../slm/model_zoo/gpt-3/external_ops/
python setup.py install
cd -

# install fast_dataindex
cd ../llm/auto_parallel/qwen
python -m pip install fast_dataindex

# download data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
mkdir data
mv llama_openwebtext_100k_ids.npy ./data
mv llama_openwebtext_100k_idx.npz ./data

# mv pretrain_config
rm -rf pretrain_config_*
cp -r ../../../tests/test_tipc/static/auto_parallel/qwen/pretrain_config_* ./
