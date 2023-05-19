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

from paddlenlp import SimpleServer, Taskflow

# The schema changed to your defined schema
schema = ["病情诊断", "治疗方案", "病因分析", "指标解读", "就医建议", "疾病表述", "后果表述", "注意事项", "功效作用", "医疗费用", "其他"]
# The task path changed to your best model path
utc = Taskflow(
    "zero_shot_text_classification", model="utc-base", task_path="../../checkpoint/model_best/plm", schema=schema
)
# If you want to define the finetuned utc service
app = SimpleServer()
app.register_taskflow("taskflow/utc", utc)
