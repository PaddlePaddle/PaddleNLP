# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

MODEL_DIR=../pruning/pruned_models/

for task in AFQMC TNEWS IFLYTEK CMNLI OCNLI CLUEWSC2020 CSL
do
    python quant_post.py --task_name ${task} --input_dir ${MODEL_DIR}/${task}/0.75/sub_static
done
