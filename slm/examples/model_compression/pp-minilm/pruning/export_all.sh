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

MODEL_PATH=pruned_models

for TASK_NAME in AFQMC TNEWS IFLYTEK CMNLI OCNLI CLUEWSC2020 CSL

do
    python export_model.py --model_type ppminilm \
    --model_name_or_path ${MODEL_PATH}/${TASK_NAME}/0.75/best_model \
    --sub_model_output_dir ${MODEL_PATH}/${TASK_NAME}/0.75/sub/  \
    --static_sub_model ${MODEL_PATH}/${TASK_NAME}/0.75/sub_static/float  \
    --n_gpu 1 --width_mult 0.75

done
