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

python -m pip install -r ../requirements.txt
python -m pip install -r ../requirements-dev.txt
export PYTHONPATH=../:${PYTHONPATH}
echo ${PYTHONPATH}


# install fused_ln custom ops
cd ../slm/model_zoo/gpt-3/external_ops/
python setup.py install
cd -

# install paddlenlp_ops
# cd ../csrc/
# python setup_cuda.py install
# cd -

cd ../llm
cp -r ../tests/test_tipc/llm/llama2/benchmark_common/benchmark_json ./

wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/paddle_data.tar.gz
tar zxvf paddle_data.tar.gz && rm -rf paddle_data.tar.gz