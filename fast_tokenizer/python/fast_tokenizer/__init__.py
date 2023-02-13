# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "1.0.2"

import os
import platform
import sys
from typing import Dict, List, Tuple, Union

from . import core_tokenizers as C
from .c_wrap import *

try:
    current_path = os.path.abspath(os.path.dirname(__file__))
    if os.name == "nt":
        third_lib_path = current_path + os.sep + "libs"
        # Will load shared library from 'path' on windows
        os.environ["path"] = current_path + ";" + third_lib_path + ";" + os.environ["path"]
        sys.path.insert(0, third_lib_path)
        # Note: from python3.8, PATH will not take effect
        # https://github.com/python/cpython/pull/12302
        # Use add_dll_directory to specify dll resolution path
        if sys.version_info[:2] >= (3, 8):
            os.add_dll_directory(third_lib_path)
except ImportError as e:
    if os.name == "nt":
        executable_path = os.path.abspath(os.path.dirname(sys.executable))
        raise ImportError(
            """NOTE: You may need to run \"set PATH=%s;%%PATH%%\"
        if you encounters \"DLL load failed\" errors. If you have python
        installed in other directory, replace \"%s\" with your own
        directory. The original error is: \n %s"""
            % (executable_path, executable_path, str(e))
        )
    else:
        raise ImportError(
            """NOTE: You may need to run \"export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\"
        if you encounters \"libmkldnn.so not found\" errors. If you have python
        installed in other directory, replace \"/usr/local/lib\" with your own
        directory. The original error is: \n"""
            + str(e)
        )
except Exception as e:
    raise e

from . import decoders, models, normalizers, postprocessors, pretokenizers
from .tokenizers_impl import (
    ClipFastTokenizer,
    ErnieFastTokenizer,
    SentencePieceBPEFastTokenizer,
)
