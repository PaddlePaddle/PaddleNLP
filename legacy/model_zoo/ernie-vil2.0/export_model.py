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

from paddlenlp.transformers import ErnieViLModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = ErnieViLModel.from_pretrained(args.model_path)
    # Switch to eval model
    model.eval()
    # Save text encoder model
    # Convert to static graph with specific input description
    static_model = paddle.jit.to_static(
        model.get_text_features,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        ],
    )

    # Save converted static graph model
    paddle.jit.save(static_model, os.path.join(args.output_path, "get_text_features"))
    # Save image encoder model
    static_model = paddle.jit.to_static(
        model.get_image_features,
        input_spec=[
            paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype="float32"),  # pixel_values
        ],
    )

    # Save converted static graph model
    paddle.jit.save(static_model, os.path.join(args.output_path, "get_image_features"))


if __name__ == "__main__":
    main()
