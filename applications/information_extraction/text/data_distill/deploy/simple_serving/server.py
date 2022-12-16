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
schema = {"武器名称": ["产国", "类型", "研发单位"]}
# The task path changed to your best model path
uie = Taskflow(
    "information_extraction", model="uie-data-distill-gp", schema=schema, task_path="../../checkpoint/model_best/"
)
# If you want to define the finetuned uie service
app = SimpleServer()
app.register_taskflow("taskflow/uie", uie)
