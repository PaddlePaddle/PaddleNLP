# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.mergekit import MergeConfig, MergeModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path0", default=None, required=True, type=str, help="The directory of model.")
    parser.add_argument("--model_name_or_path1", default=None, required=True, type=str, help="The directory of model.")
    parser.add_argument(
        "--model_name_or_path_base", default=None, required=True, type=str, help="The directory of model."
    )
    parser.add_argument("--output_path", default=None, type=str, required=True, help="The directory of saved model ")

    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "npu", "cpu"],
        help="Device for selecting for merging lora weights, currently only supports gpu/npu/cpu.",
    )
    parser.add_argument("--merge_type", default="slerp", type=str, help="The type of merge strategy.")
    parser.add_argument("--sparsify_type", default=None, type=str, help="The type of sparsify strategy.")
    parser.add_argument(
        "--dot_threshold",
        default=0.99,
        type=float,
        help="Threshold for considering the two vectors as colinear.(Used in slerp)",
    )
    parser.add_argument("--scaling", default=False, type=bool, help="Whether to scale the weights.")
    parser.add_argument("--normalize", default=True, type=bool, help="Whether to normalize the weights.")
    parser.add_argument("--drop_rate", default=0.7, type=float, help="Drop rate for the merge.")
    parser.add_argument("--della_rate", default=0.2, type=float, help="Della rate for the merge.")
    parser.add_argument(
        "--tensor_type", default="np", type=str, help="Tensor type to use for the merge. Choose np or pd"
    )
    parser.add_argument("--n_process", default=1, type=int, help="Number of processes to use for the merge.")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="Data type to use for the merge.")
    parser.add_argument("--linear_ratio", default=0.5, type=float, help="Linear merge ratio.")
    parser.add_argument("--merge_preifx", default="model", type=str, help="Prefix name: model or master_weights")
    return parser.parse_args()


def merge_model():
    args = parse_arguments()
    merge_config = MergeConfig(
        merge_type=args.merge_type,
        linear_ratio=args.linear_ratio,
        n_process=args.n_process,
        dtype=args.dtype,
        device=args.device,
        merge_preifx=args.merge_preifx,
    )
    mergekit = MergeModel(merge_config)
    model_path0 = args.model_name_or_path0
    model_path1 = args.model_name_or_path1
    base_path = args.model_name_or_path_base
    output_path = args.output_path

    mergekit.merge_model(model_path0, model_path1, output_path, base_path)
    merge_config.save_pretrained(output_path)


if __name__ == "__main__":
    merge_model()
