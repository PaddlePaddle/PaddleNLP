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

import paddle
import paddle.nn as nn
from paddle.static import InputSpec

from paddlenlp.transformers import ElectraForTotalPretraining, ElectraDiscriminator, ElectraGenerator, ElectraModel
from paddlenlp.transformers import ElectraForSequenceClassification, ElectraTokenizer

MODEL_CLASSES = {"electra": (ElectraForTotalPretraining, ElectraTokenizer), }


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
    input_model_file = os.path.join(args.input_model_dir,
                                    "model_state.pdparams")
    print(
        "load ElectraForTotalPreTraining model to get static model : %s \nmodel md5sum : %s"
        % (input_model_file, get_md5sum(input_model_file)))
    # depart total_pretraining_model to generator and discriminator state_dict
    total_pretraining_model = paddle.load(input_model_file)
    discriminator_state_dict = {}
    total_keys = []

    if all((s.startswith("generator") or s.startswith("discriminator"))
           for s in total_pretraining_model.keys()):
        print("we are load total electra model")
        num_keys = 0
        for key in total_pretraining_model.keys():
            new_key = None
            if "discriminator." in key:
                new_key = key.replace("discriminator.", "", 1)
                discriminator_state_dict[new_key] = total_pretraining_model[key]
            num_keys += 1
        print("total electra keys : ", num_keys)
    elif "discriminator_predictions.dense.weight" in total_pretraining_model:
        print("we are load discriminator model")
        discriminator_state_dict = total_pretraining_model
    else:
        print(
            "the model file : %s may not be electra pretrained model, please check"
            % input_model_file)
        exit(1)

    discriminator_model = ElectraDiscriminator(
        ElectraModel(**ElectraForTotalPretraining.pretrained_init_configuration[
            args.model_name + "-discriminator"]))
    discriminator_model.set_state_dict(
        discriminator_state_dict, use_structured_name=True)
    print("total discriminator keys : ", len(discriminator_state_dict))

    # save static model to disk
    paddle.jit.save(
        layer=discriminator_model,
        path=os.path.join(args.output_model_dir, args.model_name),
        input_spec=[InputSpec(
            shape=[None, None], dtype='int64')])
    print("save electra inference model success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model_dir",
        required=True,
        default=None,
        help="Directory for storing Electra pretraining model")
    parser.add_argument(
        "--output_model_dir",
        required=True,
        default=None,
        help="Directory for output Electra inference model")
    parser.add_argument(
        "--model_name",
        default="electra-small",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    args, unparsed = parser.parse_known_args()
    main()
