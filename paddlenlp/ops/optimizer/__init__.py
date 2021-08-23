# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.utils.cpp_extension import load

from .adamw import AdamW
from .AdamwOptimizer import AdamwOptimizer
from .adamwdl import AdamWDL


def _jit_compile():
    try:
        load(
            name="custom_jit_ops",
            sources=[
                os.path.join(os.path.dirname(__file__), x)
                for x in [
                    "adamw.cc",
                    "adamw.cu",
                ]
            ])
        return True
    except RuntimeError as e:
        import sys
        sys.stderr.write(str(e) + "\n\n")
        sys.stderr.write(
            '''Warning with compile custom ops: compile custom adamw op failed. \nIf you do not use custom ops, please ignore this warning! \n\n'''
        )
        return False


__all__ = [
    '_jit_compile',
    'AdamW',
    'AdamwOptimizer',
    'AdamWDL',
]
