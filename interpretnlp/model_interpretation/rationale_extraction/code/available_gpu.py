#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##########################################################
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved #
##########################################################

"""print available_gpu id, using nvgpu

Filname: available_gpu.py
Authors: ZhangAo(@baidu.com)
Date: 2019-12-05 15:51:32
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
        arg_parser = ArgumentParser(description="print available_gpu id, using nvgpu")
        arg_parser.add_argument("-b", "--best", default=None, type=int, help="output best N")
        args = arg_parser.parse_args()

        if args.best is not None:
            gpus = sorted(nvgpu.gpu_info(), key=lambda x: (x['mem_used'], x['index']))
            ids = [x['index'] for x in gpus]
            print(','.join(ids[:args.best]))
        else:
            print(','.join(nvgpu.available_gpus()))

    except Exception as e:
        traceback.print_exc()
        exit(-1)

