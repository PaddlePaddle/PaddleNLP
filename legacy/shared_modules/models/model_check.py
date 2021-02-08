#encoding=utf8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import paddle
import paddle.fluid as fluid


def check_cuda(use_cuda, err = \
    "\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
    Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
                                                                                                                     ):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    try:
        if use_cuda == True and fluid.is_compiled_with_cuda() == False:
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def check_version():
    """
        Log error and exit when the installed version of paddlepaddle is
        not satisfied.
        """
    err = "PaddlePaddle version 1.6 or higher is required, " \
        "or a suitable develop version is satisfied as well. \n" \
        "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        print(err)
        sys.exit(1)


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.6 or higher is required, " \
        "or a suitable develop version is satisfied as well. \n" \
        "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        print(err)
        sys.exit(1)


if __name__ == "__main__":
    check_cuda(True)

    check_cuda(False)

    check_cuda(True, "This is only for testing.")
