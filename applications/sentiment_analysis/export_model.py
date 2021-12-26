# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse
import paddle
from paddlenlp.transformers import SkepModel, ErnieModel
from extraction.model import SkepForTokenClassification
from classification.model import SkepForSequenceClassification
from pp_minilm.model import PPMiniLMForSequenceClassification

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="extraction", choices=["extraction", "classification", "pp_minilm"], help="The model type that you wanna export.")
    parser.add_argument("--base_model_name", type=str, default="skep_ernie_1.0_large_ch", help="The base model of experiment, skep or ppminilm")
    parser.add_argument("--model_path", type=str, default=None, help="The path of model that you want to load.")
    parser.add_argument("--save_path", type=str, default=None, help="The path of the exported static model.")
    args = parser.parse_args()
    # yapf: enbale

    # load model with saved state_dict
    if args.model_type == "extraction":
        skep = SkepModel.from_pretrained(args.base_model_name)
        model = SkepForTokenClassification(skep, num_classes=5)
    elif args.model_type == "classification":
        skep = SkepModel.from_pretrained(args.base_model_name)
        model = SkepForSequenceClassification(skep, num_classes=2)
    else:
        ppminilm = ErnieModel.from_pretrained(args.base_model_name)
        model = PPMiniLMForSequenceClassification(ppminilm, num_classes=2)

    loaded_state_dict = paddle.load(args.model_path)
    model.load_dict(loaded_state_dict)
    print(f"Loaded parameters from {args.model_path}")

    model.eval()
    # convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # token_type_ids
        ])

    # save to static model
    paddle.jit.save(model, args.save_path)
    print(f"static {args.model_type} model has been to {args.save_path}")
