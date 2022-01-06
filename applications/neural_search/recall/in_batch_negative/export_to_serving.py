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
import paddle_serving_client.io as serving_io
# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True,
                    default='./checkpoint/model_900/model_state.pdparams', help="The path to model parameters to be loaded.")
parser.add_argument("--model_filename", type=str, required=True,
                    default='inference.get_pooled_embedding.pdmodel', help="The path to model parameters to be loaded.")
parser.add_argument("--params_filename", type=str, required=True,
                    default='inference.get_pooled_embedding.pdiparams', help="The path to model parameters to be loaded.")
parser.add_argument("--server_path", type=str, default='./serving_server',
                    help="The path of server parameter in static graph to be saved.")
parser.add_argument("--client_path", type=str, default='./serving_client',
                    help="The path of client parameter in static graph to be saved.")
parser.add_argument("--feed_alias_names", type=str, default=None,
                    help='set alias names for feed vars, split by comma \',\', you should run --show_proto to check the number of feed vars')
parser.add_argument("--fetch_alias_names", type=str, default=None,
                    help='set alias names for feed vars, split by comma \',\', you should run --show_proto to check the number of fetch vars')
parser.add_argument("--show_proto", type=bool, default=True,
                    help='If yes, you can preview the proto and then determine your feed var alias name and fetch var alias name.')
# yapf: enable

if __name__ == "__main__":
    args = parser.parse_args()
    dirname = args.params_path
    serving_io.inference_model_to_serving(
        dirname,
        serving_server=args.server_path,
        serving_client=args.client_path,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        show_proto=args.show_proto,
        feed_alias_names=args.feed_alias_names,
        fetch_alias_names=args.fetch_alias_names)
