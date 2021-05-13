#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##########################################################
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved #
##########################################################
"""model io utils

Filname: io.py
Authors: ZhangAo(@baidu.com)
Date: 2021-01-18 14:44:58
"""

import sys
import os
import traceback
import logging
import json

import paddle


def init_ernie_model(model_class, model_dir):
    """init ernie model from static graph checkpoint

    Args:
        model_class (TYPE): NULL
        model_dir (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    with open(os.path.join(model_dir, 'ernie_config.json')) as ifs:
        config = json.load(ifs)

    state = paddle.static.load_program_state(os.path.join(model_dir, 'params'))
    ernie = model_class(config, name='')
    ernie.set_dict(state, use_structured_name=False)
    return ernie, config['hidden_size']


def save(model, optimzer, save_path):
    """

    Args:
        model (TYPE): NULL
        optimzer (TYPE): NULL
        save_path (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
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
