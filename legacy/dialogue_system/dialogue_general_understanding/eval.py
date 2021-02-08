# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"""evaluation"""

import os
import sys

from dgu.evaluation import evaluate
from dgu.utils.configure import PDConfig


def do_eval(args):

    task_name = args.task_name.lower()
    reference = args.evaluation_file
    predicitions = args.output_prediction_file

    evaluate(task_name, predicitions, reference)


if __name__ == "__main__":
    import paddle
    paddle.enable_static()

    args = PDConfig(yaml_file="./data/config/dgu.yaml")
    args.build()

    do_eval(args)
