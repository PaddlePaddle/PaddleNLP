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
cd ../llm/
python -m pip install fast_dataindex

# download data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
mkdir data
mv llama_openwebtext_100k_ids.npy ./data
mv llama_openwebtext_100k_idx.npz ./data

# install etcd
pip install httpx etcd3 protobuf==3.20.0 --force-reinstall
ip_lists=($(echo $TRAINER_INSTANCES | tr ',' ' '))
master_ip=${ip_lists[0]}
rank=$PADDLE_TRAINER_ID
echo $master_ip $rank
#多机任务在每台机器上都启动服务，保证同步，否则多机运行会报错
net=$(netstat -anp | grep ":2379" | grep "LISTEN")
if [ ${#net} == 0 ]; then
    nohup etcd -data-dir ~/data.etcd -advertise-client-urls http://0.0.0.0:2379 -listen-client-urls http://0.0.0.0:2379 &
    ps -ef |grep etcd
fi  

# mv autoconfig
rm -rf auto_config_*
cp -r ../tests/test_tipc/dygraph/hybrid_parallelism/llama2/auto_config_* ./
