# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from reprod_log import ReprodLogger

if __name__ == "__main__":
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()

    data_1 = np.random.rand(4, 64, 768).astype(np.float32)
    data_2 = np.random.rand(4, 64, 768).astype(np.float32)

    reprod_log_1.add("demo_test_1", data_1)
    reprod_log_1.add("demo_test_2", data_1)
    reprod_log_1.save("result_1.npy")

    reprod_log_2.add("demo_test_1", data_1)
    reprod_log_2.add("demo_test_2", data_2)
    reprod_log_2.save("result_2.npy")
