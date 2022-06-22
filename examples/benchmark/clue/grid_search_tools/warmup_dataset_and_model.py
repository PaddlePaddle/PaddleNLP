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
from paddlenlp.utils.log import logger

model_name_or_path = sys.argv[1]

# CLUE classification dataset warmup
logger.info("Download model and data for CLUE classification tasks.")
for task in [
        "afqmc", "tnews", "iflytek", "ocnli", "cmnli", "cluewsc2020", "csl"
]:
    load_dataset("clue", task, splits=("train", "dev", "test"))

# Downloads HF dataset
from datasets import load_dataset

load_dataset("clue", "chid")
load_dataset("clue", "cmrc2018")
load_dataset("clue", "c3")

# HF dataset process and cache
logger.info(
    "Data process for CHID tasks, and this will take some time. If cache exists, this will skip."
)
status = os.system(
    f"python ../mrc/run_chid.py --do_train --max_steps 0 --model_name_or_path {model_name_or_path} --batch_size 1 --gradient_accumulation_steps 1"
)
assert status == 0, "Please make sure clue dataset CHID has been preprocessed successfully."
logger.info("Data process for CMRC2018 tasks. If cache exists, this will skip.")
status = os.system(
    f"python ../mrc/run_cmrc2018.py --do_train --max_steps 0 --model_name_or_path {model_name_or_path} --batch_size 1 --gradient_accumulation_steps 1"
)
assert status == 0, "Please make sure clue dataset CMRC2018 has been preprocessed successfully."
logger.info("Data process for C3 tasks. If cache exists, this will skip.")
status = os.system(
    f"python ../mrc/run_c3.py --do_train --max_steps 0 --model_name_or_path {model_name_or_path} --batch_size 1 --gradient_accumulation_steps 1"
)
assert status == 0, "Please make sure clue dataset C3 has been preprocessed successfully."
