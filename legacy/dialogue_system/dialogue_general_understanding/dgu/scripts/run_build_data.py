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
import shutil
import sys
import os

from build_atis_dataset import ATIS
from build_dstc2_dataset import DSTC2
from build_mrda_dataset import MRDA
from build_swda_dataset import SWDA

if __name__ == "__main__":
    task_name = sys.argv[1]
    task_name = task_name.lower()

    if task_name not in ['swda', 'mrda', 'atis', 'dstc2', 'udc']:
        print("task name error: we support [swda|mrda|atis|dstc2|udc]")
        exit(1)

    if task_name == 'swda':
        swda_inst = SWDA()
        swda_inst.main()
    elif task_name == 'mrda':
        mrda_inst = MRDA()
        mrda_inst.main()
    elif task_name == 'atis':
        atis_inst = ATIS()
        atis_inst.main()
        shutil.copyfile("../../data/input/data/atis/atis_slot/test.txt",
                        "../../data/input/data/atis/atis_slot/dev.txt")
        shutil.copyfile("../../data/input/data/atis/atis_intent/test.txt",
                        "../../data/input/data/atis/atis_intent/dev.txt")
    elif task_name == 'dstc2':
        dstc_inst = DSTC2()
        dstc_inst.main()
    else:
        exit(0)
