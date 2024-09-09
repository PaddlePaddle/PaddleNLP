# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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
import subprocess

path = os.path.abspath(os.path.dirname(__file__))


def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""
    import sys

    excutable = sys.executable
    ret = subprocess.run(["make", "-C", path, f"PYTHON_BIN={excutable}"])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        sys.exit(1)
