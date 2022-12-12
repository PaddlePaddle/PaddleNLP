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

import argparse
import os

import paddle

from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default='./checkpoint/', help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./export', help="The path of model parameter in static graph to be saved.")
parser.add_argument("--task_type", type=str, default='seq_cls', choices=["seq_cls", "token_cls"], help="The task type of ernie 3.0")
args = parser.parse_args()
# yapf: enable

model_class_dict = {"seq_cls": AutoModelForSequenceClassification, "token_cls": AutoModelForTokenClassification}

if __name__ == "__main__":
    model_class = model_class_dict[args.task_type]
    model = model_class.from_pretrained(args.params_path)
    model.eval()
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
    ]
    # Convert to static graph with specific input description
    model = paddle.jit.to_static(model, input_spec=input_spec)

    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
