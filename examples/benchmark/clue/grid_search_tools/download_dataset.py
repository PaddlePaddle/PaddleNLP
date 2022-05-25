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

from paddlenlp.datasets import load_dataset
for task in [
        "afqmc", "tnews", "iflytek", "ocnli", "cmnli", "cluewsc2020", "csl"
]:
    load_dataset("clue", task, splits=("train", "dev", "test"))

from datasets import load_dataset
load_dataset("clue", "cmrc2018", split=("train", "validation", "test"))
load_dataset("clue", "chid", split=("train", "validation", "test"))
load_dataset("clue", "c3", split=("train", "validation", "test"))
