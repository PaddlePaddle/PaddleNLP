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

import argparse

from src.tot.methods.bfs import solve
from src.tot.tasks.game24 import Game24Task

args = argparse.Namespace(
    backend="llama-2-7b-chat",
    temperature=0.6,
    task="game24",
    naive_run=False,
    prompt_sample=None,
    method_generate="propose",
    method_evaluate="value",
    method_select="greedy",
    n_generate_sample=1,
    n_evaluate_sample=3,
    n_select_sample=5,
)

task = Game24Task()
ys, infos = solve(args, task, 900)
print(ys[0])
print(infos)
