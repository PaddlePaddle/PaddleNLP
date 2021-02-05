#encoding="utf8"
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
import os
import struct
import logging
import argparse
import numpy as np
import collections

import paddle.fluid as F
from paddle.fluid.proto import framework_pb2

log = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
log.addHandler(console)
log.setLevel(logging.DEBUG)


def gen_arr(data, dtype):
    num = len(data) // struct.calcsize(dtype)
    arr = struct.unpack('%d%s' % (num, dtype), data)
    return arr


def parse(filename):
    with open(filename, 'rb') as f:
        read = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        _, = read('I')  # version
        lodsize, = read('Q')
        if lodsize != 0:
            log.warning('shit, it is LOD tensor!!! skipped!!')
            return None
        _, = read('I')  # version
        pbsize, = read('i')
        data = f.read(pbsize)
        proto = framework_pb2.VarType.TensorDesc()
        proto.ParseFromString(data)
        log.info('type: [%s] dim %s' % (proto.data_type, proto.dims))
        if proto.data_type == framework_pb2.VarType.FP32:
            arr = np.array(
                gen_arr(f.read(), 'f'), dtype=np.float32).reshape(proto.dims)
        elif proto.data_type == framework_pb2.VarType.INT64:
            arr = np.array(
                gen_arr(f.read(), 'q'), dtype=np.int64).reshape(proto.dims)
        elif proto.data_type == framework_pb2.VarType.INT32:
            arr = np.array(
                gen_arr(f.read(), 'i'), dtype=np.int32).reshape(proto.dims)
        elif proto.data_type == framework_pb2.VarType.INT8:
            arr = np.array(
                gen_arr(f.read(), 'B'), dtype=np.int8).reshape(proto.dims)
        elif proto.data_type == framework_pb2.VarType.FP16:
            arr = np.array(
                gen_arr(f.read(), 'H'),
                dtype=np.uint16).view(np.float16).reshape(proto.dims)
        else:
            raise RuntimeError('Unknown dtype %s' % proto.data_type)

        return arr


def show(arr):
    print(repr(arr))


def dump(arr, path):
    path = os.path.join(args.to, path)
    log.info('dump to %s' % path)
    try:
        os.makedirs(os.path.dirname(path))
    except:
        pass
    # np.save(open(path, 'wb'), arr)  
    np.save(path, arr)


def list_dir(dir_or_file):
    if os.path.isfile(dir_or_file):
        return [dir_or_file]
    else:
        return [
            os.path.join(i, kk) for i, _, k in os.walk(dir_or_file) for kk in k
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['show', 'dump'], type=str)
    parser.add_argument('file_or_dir', type=str)
    parser.add_argument('-t', "--to", type=str, default=None)
    parser.add_argument('-v', "--verbose", action='store_true')
    args = parser.parse_args()

    files = list_dir(args.file_or_dir)
    parsed_arr = map(parse, files)
    if args.mode == 'show':
        for arr in parsed_arr:
            if arr is not None:
                show(arr)
    elif args.mode == 'dump':
        if args.to is None:
            raise ValueError('--to dir_name not specified')
        for arr, path in zip(parsed_arr, files):
            if arr is not None:
                dump(arr, path.replace(args.file_or_dir, ''))
