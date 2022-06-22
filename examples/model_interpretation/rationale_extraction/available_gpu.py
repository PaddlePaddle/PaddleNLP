#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# See the License for the specific l
"""print available_gpu id, using nvgpu
"""

import sys
import os
import traceback
import logging
import nvgpu

logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s: %(asctime)s %(filename)s'
                    ' [%(funcName)s:%(lineno)d][%(process)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename=None,
                    filemode='a')

if __name__ == "__main__":
    from argparse import ArgumentParser
    try:
        arg_parser = ArgumentParser(
            description="print available_gpu id, using nvgpu")
        arg_parser.add_argument("-b",
                                "--best",
                                default=None,
                                type=int,
                                help="output best N")
        args = arg_parser.parse_args()

        if args.best is not None:
            gpus = sorted(nvgpu.gpu_info(),
                          key=lambda x: (x['mem_used'], x['index']))
            ids = [x['index'] for x in gpus]
            print(','.join(ids[:args.best]))
        else:
            print(','.join(nvgpu.available_gpus()))

    except Exception as e:
        traceback.print_exc()
        exit(-1)
