# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
#from collections import namedtuple
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import argparse
import json

import paddle
import paddle.nn as nn
from paddle.static import InputSpec

from paddlenlp.transformers import ElectraForTotalPretraining, ElectraDiscriminator, ElectraGenerator, ElectraModel
from paddlenlp.transformers import ElectraForSequenceClassification, ElectraTokenizer


def get_md5sum(file_path):
    md5sum = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            md5_obj = hashlib.md5()
            md5_obj.update(f.read())
            hash_code = md5_obj.hexdigest()
        md5sum = str(hash_code).lower()
    return md5sum


def main():
    # check and load config
    with open(os.path.join(args.input_model_dir, "model_config.json"),
              'r') as f:
        config_dict = json.load(f)
        num_classes = config_dict['num_classes']
    if num_classes is None or num_classes <= 0:
        print("%s/model_config.json may not be right, please check" %
              args.input_model_dir)
        exit(1)

    # check and load model
    input_model_file = os.path.join(args.input_model_dir,
                                    "model_state.pdparams")
    print("load model to get static model : %s \nmodel md5sum : %s" %
          (input_model_file, get_md5sum(input_model_file)))
    model_state_dict = paddle.load(input_model_file)

    if all((s.startswith("generator") or s.startswith("discriminator"))
           for s in model_state_dict.keys()):
        print(
            "the model : %s is electra pretrain model, we need fine-tuning model to deploy"
            % input_model_file)
        exit(1)
    elif "discriminator_predictions.dense.weight" in model_state_dict:
        print(
            "the model : %s is electra discriminator model, we need fine-tuning model to deploy"
            % input_model_file)
        exit(1)
    elif "classifier.dense.weight" in model_state_dict:
        print("we are load glue fine-tuning model")
        model = ElectraForSequenceClassification.from_pretrained(
            args.input_model_dir, num_classes=num_classes)
        print("total model layers : ", len(model_state_dict))
    else:
        print("the model file : %s may not be fine-tuning model, please check" %
              input_model_file)
        exit(1)

    # save static model to disk
    paddle.jit.save(layer=model,
                    path=os.path.join(args.output_model_dir, args.model_name),
                    input_spec=[InputSpec(shape=[None, None], dtype='int64')])
    print("save electra inference model success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_dir",
                        required=True,
                        default=None,
                        help="Directory for storing Electra pretraining model")
    parser.add_argument("--output_model_dir",
                        required=True,
                        default=None,
                        help="Directory for output Electra inference model")
    parser.add_argument("--model_name",
                        default="electra-deploy",
                        type=str,
                        help="prefix name of output model and parameters")
    args, unparsed = parser.parse_known_args()
    main()
