# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
'''
Script for downloading training data.
'''
import os
import sys
import shutil
import argparse
import tempfile
import urllib
import tarfile
import io
if sys.version_info >= (3, 0):
    import urllib.request
import zipfile

URLLIB = urllib
if sys.version_info >= (3, 0):
    URLLIB = urllib.request

TASKS = ['ptb', 'swda']
TASK2PATH = {
    'ptb': 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz',
    'swda': 'https://baidu-nlp.bj.bcebos.com/TextGen/swda.tar.gz'
}


def un_tar(tar_name, dir_name):
    try:
        t = tarfile.open(tar_name)
        t.extractall(path=dir_name)
        return True
    except Exception as e:
        print(e)
        return False


def download_and_extract(task, data_path):
    print('Downloading and extracting %s...' % task)
    data_file = os.path.join(data_path, TASK2PATH[task].split('/')[-1])
    URLLIB.urlretrieve(TASK2PATH[task], data_file)
    un_tar(data_file, data_path)
    os.remove(data_file)
    if task == 'ptb':
        src_dir = os.path.join(data_path, 'simple-examples')
        dst_dir = os.path.join(data_path, 'ptb')
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        shutil.copy(os.path.join(src_dir, 'data/ptb.train.txt'), dst_dir)
        shutil.copy(os.path.join(src_dir, 'data/ptb.valid.txt'), dst_dir)
        shutil.copy(os.path.join(src_dir, 'data/ptb.test.txt'), dst_dir)
        shutil.rmtree(src_dir)
    print('\tCompleted!')


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_dir',
        help='directory to save data to',
        type=str,
        default='data')
    parser.add_argument(
        '-t',
        '--task',
        help='tasks to download data for as a comma separated string',
        type=str,
        default='ptb')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    download_and_extract(args.task, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
