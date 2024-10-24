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

wget https://paddlenlp.bj.bcebos.com/data/model_interpretation.tar
wait
tar -xvf model_interpretation.tar
wait
mv ./model_interpretation/vocab.char ./task/similarity/simnet/
mv ./model_interpretation/vocab_QQP ./task/similarity/simnet/
mv ./model_interpretation/simnet_vocab.txt ./task/similarity/simnet/

mv ./model_interpretation/vocab.sst2_train ./task/senti/rnn/
mv ./model_interpretation/vocab.txt ./task/senti/rnn