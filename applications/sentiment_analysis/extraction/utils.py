# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


import hashlib
import paddle
import random
import numpy as np

def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def compute_md5(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    file_md5 = hashlib.md5(data).hexdigest()
    print(file_md5)
