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
import os
import hashlib
import argparse

import paddle
import paddle.nn as nn

#from paddlenlp.transformers import ElectraForTotalPretraining, ElectraDiscriminator, ElectraGenerator, ElectraModel
#from paddlenlp.transformers import ElectraTokenizer
#
#MODEL_CLASSES = {"electra": (ElectraForTotalPretraining, ElectraTokenizer), }


def get_md5sum(file_path):
    md5sum = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            md5_obj = hashlib.md5()
            md5_obj.update(f.read())
            hash_code = md5_obj.hexdigest()
        md5sum = str(hash_code).lower()
    return md5sum


def main(args):
    pretraining_model = os.path.join(args.model_dir, "model_state.pdparams")
    if os.path.islink(pretraining_model):
        print("%s already contain fine-tuning model, pleace check" %
              args.model_dir)
        exit(0)
    print(
        "load Electra pretrain model to get generator/discriminator model : %s \nmodel md5sum : %s"
        % (pretraining_model, get_md5sum(pretraining_model)))
    # depart total_pretraining_model to generator and discriminator state_dict
    total_pretraining_model = paddle.load(pretraining_model)
    generator_state_dict = {}
    discriminator_state_dict = {}
    total_keys = []
    num_keys = 0
    for key in total_pretraining_model.keys():
        new_key = None
        if "generator." in key:
            new_key = key.replace("generator.", "", 1)
            generator_state_dict[new_key] = total_pretraining_model[key]
        if "discriminator." in key:
            new_key = key.replace("discriminator.", "", 1)
            discriminator_state_dict[new_key] = total_pretraining_model[key]
        num_keys += 1
    print("total electra keys : ", num_keys)
    print("total generator keys : ", len(generator_state_dict))
    print("total discriminator keys : ", len(discriminator_state_dict))

    # save generator and discriminator model to disk
    paddle.save(generator_state_dict,
                os.path.join(args.model_dir, args.generator_output_file))
    paddle.save(discriminator_state_dict,
                os.path.join(args.model_dir, args.discriminator_output_file))
    print("save generator and discriminator model success")
    os.rename(pretraining_model,
              os.path.join(args.model_dir, "pretrain_model_state.pdparams"))
    os.symlink(args.discriminator_output_file,
               os.path.join(args.model_dir, "model_state.pdparams"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        default=None,
        help="Directory of storing ElectraForTotalPreTraining model")
    parser.add_argument("--generator_output_file",
                        default='generator_for_ft.pdparams',
                        help="Electra generator model for fine-tuning")
    parser.add_argument("--discriminator_output_file",
                        default='discriminator_for_ft.pdparams',
                        help="Electra discriminator model for fine-tuning")
    args, unparsed = parser.parse_known_args()
    main(args)
