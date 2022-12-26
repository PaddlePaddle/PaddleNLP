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
schema = [{"评价维度": ["观点词", "情感倾向[正向,负向,未提及]"]}]
# define taskflow to perform sentiment analysis
senta = Taskflow("sentiment_analysis", schema=schema, model="uie-senta-base")
# define your server
app = SimpleServer()
app.register_taskflow("taskflow/senta", senta)
