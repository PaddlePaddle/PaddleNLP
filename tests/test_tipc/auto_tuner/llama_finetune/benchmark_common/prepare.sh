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

cd ../llm/
rm -rf data
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz
tar -zxvf AdvertiseGen.tar.gz && rm -rf AdvertiseGen.tar.gz

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
  if [ $rank == 0 ]; then
    net=$(netstat -anp | grep :2379 | grep "LISTEN")
    if [ ${#net} == 0 ]; then
        apt-get install -y --allow-downgrades etcd
        nohup etcd -data-dir ~/data.etcd -advertise-client-urls  http://0.0.0.0:2379 -listen-client-urls http://0.0.0.0:2379 &
        ps -ef |grep etcd
    fi  
  else
      sleep 5
  fi
  sleep 5
fi
