# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import tarfile
import shutil
import urllib
import sys
import io
import os

URLLIB = urllib
if sys.version_info >= (3, 0):
    import urllib.request
    URLLIB = urllib.request

DATA_MODEL_PATH = {
    "DATA_PATH":
    "https://baidu-nlp.bj.bcebos.com/auto_dialogue_evaluation_dataset-1.0.0.tar.gz",
    "TRAINED_MODEL":
    "https://baidu-nlp.bj.bcebos.com/auto_dialogue_evaluation_models.3.0.0.tar.gz"
}

PATH_MAP = {'DATA_PATH': "./data/input", 'TRAINED_MODEL': './data/saved_models'}


def un_tar(tar_name, dir_name):
    try:
        t = tarfile.open(tar_name)
        t.extractall(path=dir_name)
        return True
    except Exception as e:
        print(e)
        return False


def download_model_and_data():
    print("Downloading ade data, pretrain model and trained models......")
    print("This process is quite long, please wait patiently............")
    for path in ['./data/input/data', './data/saved_models/trained_models']:
        if not os.path.exists(path):
            continue
        shutil.rmtree(path)
    for path_key in DATA_MODEL_PATH:
        filename = os.path.basename(DATA_MODEL_PATH[path_key])
        URLLIB.urlretrieve(DATA_MODEL_PATH[path_key],
                           os.path.join("./", filename))
        state = un_tar(filename, PATH_MAP[path_key])
        if not state:
            print("Tar %s error....." % path_key)
            return False
        os.remove(filename)
    return True


if __name__ == "__main__":
    state = download_model_and_data()
    if not state:
        exit(1)
    print("Downloading data and models sucess......")
