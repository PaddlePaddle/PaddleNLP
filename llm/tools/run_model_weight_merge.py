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
from paddlenlp.mergekit import MergeConfig, MergeModel

merge_config = MergeConfig(
    merge_type="della_linear",
    linear_ratio=0.5,
    n_process=2,
    dtype="bfloat16",
    device="cpu",
    merge_preifx="model",
)
mergekit = MergeModel(merge_config)
model_path0 = "/root/paddlejob/workspace/env_run/output/huxinye/model/cnn1"
model_path1 = "/root/paddlejob/workspace/env_run/output/huxinye/model/cnn2"
base_path = "/root/paddlejob/workspace/env_run/output/huxinye/model/cnn_base"
output_path = "/root/paddlejob/workspace/env_run/output/huxinye/model/cnn_merge"


mergekit.merge_model(model_path0, model_path1, output_path, base_path)
merge_config.save_pretrained(output_path)
