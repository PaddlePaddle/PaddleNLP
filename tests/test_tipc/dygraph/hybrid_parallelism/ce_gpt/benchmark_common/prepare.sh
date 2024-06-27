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
# get data
cd ../llm
rm -rf data
mkdir data
wget -O data/gpt2-en-mmap.bin https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/gpt/mmap/gpt2-en-mmap.bin
wget -O data/gpt2-en-mmap.idx https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/gpt/mmap/gpt2-en-mmap.idx
