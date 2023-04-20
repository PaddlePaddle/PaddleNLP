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
import os

import paddle


def extract_controlnet_ema_weights(model_path, output_path):
    state_dict = paddle.load(model_path, return_numpy=True)
    ema_state_dict = {}
    for k in state_dict.keys():
        if k.startswith("controlnet."):
            flat_ema_key = "model_ema." + "".join(k.split(".")[1:])
            ema_state_dict[k.replace("controlnet.", "")] = state_dict.get(flat_ema_key)
    if len(ema_state_dict) == 0:
        raise ValueError("Can not extract ema weights!")
    os.makedirs(output_path, exist_ok=True)
    paddle.save(ema_state_dict, os.path.join(output_path, "model_state.ema.pdparams"))
    print(f"Save EMA weights to {output_path} !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model_state.pdparams",
        help="model_state.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ema_controlnet",
        help="The model output path.",
    )
    args = parser.parse_args()
    extract_controlnet_ema_weights(args.model_path, args.output_path)
