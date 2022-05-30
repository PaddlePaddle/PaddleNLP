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

import sys
import os
from paddlenlp.datasets import load_dataset

model_name_or_path = sys.argv[1]

# CLUE classification dataset warmup
for task in [
        "afqmc", "tnews", "iflytek", "ocnli", "cmnli", "cluewsc2020", "csl"
]:
    load_dataset("clue", task, splits=("train", "dev", "test"))

# HF dataset warmup
status = os.system(
    'python ../mrc/run_chid.py --do_train --max_steps 1 --model_name_or_path {model_name_or_path}'
)
status = os.system(
    'python ../mrc/run_cmrc.py --do_train --max_steps 1 --model_name_or_path {model_name_or_path}'
)
status = os.system(
    'python ../mrc/run_c3.py --do_train --max_steps 1 --model_name_or_path {model_name_or_path}'
)
