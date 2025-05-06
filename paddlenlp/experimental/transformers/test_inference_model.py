# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# 后面会删掉，仅提供示例
from inference_utils import ModelArgument, PredictorArgument

from paddlenlp.experimental.transformers.inference_model import InferenceModel

predictor_args = PredictorArgument()
model_args = ModelArgument()

predictor_args.model_name_or_path = (
    "/root/paddlejob/workspace/env_run/output/gaoziyuan/paddlenllp_model/models/Qwen/Qwen2-7B"
)

# 如果需要
predictor_args.quant_type = "weight_only_int8"

inference_model = InferenceModel(predictor_args, model_args, load_model_from_ipc=True, cold_start=True)

model = inference_model.get_model()
print(model.get_name_mappings_to_training())

# 获取inference model 的 key\shape\type
inference_model.get_model_static_info()

# 获取qwen2 训练和推理的key映射关系
infer_to_train = model.get_name_mappings_to_training()

print(infer_to_train)

# keys_all = model.state_dict().keys()
# for k, v in infer_to_train.items():
#     if k not in keys_all:
#         print("missing key is :", k)
#         print("missing v is :", v)
