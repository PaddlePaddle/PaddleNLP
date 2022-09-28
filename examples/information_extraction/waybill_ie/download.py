# -*- coding: utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse

from paddle.utils.download import get_path_from_url

URL = "https://bj.bcebos.com/paddlenlp/paddlenlp/datasets/waybill.tar.gz"


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_dir',
                        help='directory to save data to',
                        type=str,
                        default='./')
    args = parser.parse_args(arguments)
    get_path_from_url(URL, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
