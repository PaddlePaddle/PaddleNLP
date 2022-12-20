# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def prepare():

    bos_link_train = "https://paddlenlp.bj.bcebos.com/datasets/tiny_summary_dataset/train.json"
    bos_link_valid = "https://paddlenlp.bj.bcebos.com/datasets/tiny_summary_dataset/valid.json"
    bos_link_test = "https://paddlenlp.bj.bcebos.com/datasets/tiny_summary_dataset/test.json"
    os.system("mkdir data")
    os.system("cd data && wget %s " % (bos_link_train))
    os.system("cd data && wget %s " % (bos_link_valid))
    os.system("cd data && wget %s " % (bos_link_test))


prepare()
