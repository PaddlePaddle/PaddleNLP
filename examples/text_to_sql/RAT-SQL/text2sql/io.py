#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
import json

import paddle


def init_ernie_model(model_class, model_dir):
    """init ernie model from static graph checkpoint
    """
    with open(os.path.join(model_dir, 'ernie_config.json')) as ifs:
        config = json.load(ifs)

    state = paddle.static.load_program_state(os.path.join(model_dir, 'params'))
    ernie = model_class(config, name='')
    ernie.set_dict(state, use_structured_name=False)
    return ernie, config['hidden_size']


def save(model, optimzer, save_path):
    try:
        paddle.save(model.state_dict(), save_path + '.pdparams')
        paddle.save(optimzer.state_dict(), save_path + '.pdopt')
    except Exception as e:
        logging.error('save model and optimzer failed. save path: %s',
                      save_path)
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    """run some simple test cases"""
    pass
