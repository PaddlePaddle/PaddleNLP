#!/bin/bash                                                                                                                                               

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
                                                                                                                                                            
# get sshd port                                                                                                                                           
sshport=$(lsof -i | grep sshd | awk '{print $9}' | sed s/\*://)                                                                                           
                                                                                                                                                            
hostfile=${TRAIN_WORKSPACE}/hostfile                                                                                                                      
hostlist=$(cat $hostfile | awk '{print $1}' | xargs)                                                                                                      
for host in ${hostlist[@]}; do                                                                                                                            
  #ssh $host "ls $PWD"                                                                                                                                    
  echo "scp $1 to $host"                                                                                                                                  
  scp -r $1 ${host}:${PWD}                                                                                                                                
done
