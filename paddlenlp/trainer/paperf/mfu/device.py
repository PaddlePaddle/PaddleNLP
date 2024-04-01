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


class AmpereConfig(object):
    def __init__(self, device):
        self.device = device
        assert device in ["40G-A100", "80G-A100", "80G-A800"]

        self.arch = "Ampere"
        self.compute_capability = "sm80"

        self.peak_flops = {
            "fp32": 19.5,  # TFLOPS
            "fp16": 78,  # TFLOPS
            "bf16": 39,  # TFLOPS
            "tf32_tensor_core": 156,  # TFLOPS
            "fp16_tensor_core": 312,  # TFLOPS
            "bf16_tensor_core": 312,  # TFLOPS
        }

        if device == "80G-A100":
            self.memory_capacity = 80  # GB
            self.memory_bandwidth = 2039  # GB/s
            self.nvlink_bandwith = 600  # GB/s
        elif device == "80G-A800":
            # SXM Version
            self.memory_capacity = 80  # GB
            self.memory_bandwidth = 2039  # GB/s
            self.nvlink_bandwith = 400  # GB/s
        else:
            self.memory_capacity = None
            self.memory_bandwidth = None
            self.nvlink_bandwith = None
