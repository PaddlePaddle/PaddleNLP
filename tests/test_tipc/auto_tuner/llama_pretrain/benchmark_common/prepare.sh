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
cd ../llm
python -m pip install fast_dataindex

rm -rf data && mkdir data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx

mv llama_openwebtext_100k.bin ./data
mv llama_openwebtext_100k.idx ./data

# mv autoconfig
rm -rf autoconfig
cp -r ../tests/test_tipc/auto_tuner/autoconfig ./

if [ -z "$1" ]; then  
  echo "单机任务"
else
  echo "多机任务, 启动etcd服务"
  pip install httpx etcd3 protobuf==3.20.0 --force-reinstall
  ip_lists=($(echo $TRAINER_INSTANCES | tr ',' ' '))
  master_ip=${ip_lists[0]}
  rank=$PADDLE_TRAINER_ID
  echo $master_ip $rank
  #多机任务在每台机器上都启动服务，保证同步，否则多机运行会报错
  net=$(netstat -anp | grep :2379 | grep "LISTEN")
  if [ ${#net} == 0 ]; then
      apt-get install -y --allow-downgrades etcd
      nohup etcd -data-dir ~/data.etcd -advertise-client-urls  http://0.0.0.0:2379 -listen-client-urls http://0.0.0.0:2379 &
      ps -ef |grep etcd
  fi  
  sleep 5
fi
