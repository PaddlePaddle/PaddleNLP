# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

python export_to_serving.py \
    --dirname "output" \
    --model_filename "inference.get_pooled_embedding.pdmodel" \
    --params_filename "inference.get_pooled_embedding.pdiparams" \
    --server_path "serving_server" \
    --client_path "serving_client" \
    --fetch_alias_names "output_embedding"
