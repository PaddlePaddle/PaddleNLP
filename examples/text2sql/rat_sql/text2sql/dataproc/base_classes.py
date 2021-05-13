#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##########################################################
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved #
##########################################################
"""define of base classes

Filname: base_classes.py
Authors: ZhangAo(@baidu.com)
Date: 2021-01-25 10:48:50
"""

import sys
import os
import traceback
import logging


class BaseInputEncoder(object):
    """Docstring for BaseInputEncoder. """

    def __init__(self):
        """init of class """
        super(BaseInputEncoder, self).__init__()

    def encode(self, inputs):
        """build inputs for model

        Args:
            inputs (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        raise NotImplementedError


if __name__ == "__main__":
    """run some simple test cases"""
    pass
