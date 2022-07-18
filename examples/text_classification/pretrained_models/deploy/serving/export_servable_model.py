# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle_serving_client.io as serving_io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_model_dir",
                        type=str,
                        default="./export/",
                        help="The directory of the inference model.")
    parser.add_argument("--model_file",
                        type=str,
                        default='inference.pdmodel',
                        help="The inference model file name.")
    parser.add_argument("--params_file",
                        type=str,
                        default='inference.pdiparams',
                        help="The input inference parameters file name.")
    return parser.parse_args()


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    feed_names, fetch_names = serving_io.inference_model_to_serving(
        dirname=args.inference_model_dir,
        serving_server="serving_server",
        serving_client="serving_client",
        model_filename=args.model_file,
        params_filename=args.params_file)
    print("model feed_names : %s" % feed_names)
    print("model fetch_names : %s" % fetch_names)
