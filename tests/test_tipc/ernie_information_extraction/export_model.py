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
import os
import paddle
from paddlenlp.transformers import AutoModelForTokenClassification
from data import load_dict

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, required=True, default='./checkpoint/model_900', help="The path to model parameters to be loaded.")
    parser.add_argument("--output_path", type=str, default='./output', help="The path of model parameter in static graph to be saved.")
    parser.add_argument("--data_dir", type=str, default="./waybill_ie/data", help="The folder where the dataset is located.")
    args = parser.parse_args()
    # yapf: enable

    # The number of labels should be in accordance with the training dataset.
    label_vocab = load_dict(os.path.join(args.data_dir, 'tag.dic'))

    model = AutoModelForTokenClassification.from_pretrained(
        args.params_path, num_classes=len(label_vocab))
    model.eval()

    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64")  # segment_ids
        ])

    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
