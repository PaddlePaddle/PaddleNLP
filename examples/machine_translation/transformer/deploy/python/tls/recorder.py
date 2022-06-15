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

import os
import time
import paddle
import psutil

from tls.benchmark_utils import PaddleInferBenchmark

import paddle
if paddle.is_compiled_with_cuda():
    import pynvml
    import GPUtil


class Recorder(object):

    def __init__(self, config, batch_size, model_name, mem_info=None):
        self.model_name = model_name
        self.config = config
        self.precision = "fp32"
        self.batch_size = batch_size

        self.use_gpu = False
        self.use_xpu = False
        self.use_cpu = False

        if config.use_gpu():
            self.place = "gpu"
            self.use_gpu = True
        elif config.use_xpu():
            self.place = "xpu"
            self.use_xpu = True
        else:
            self.place = "cpu"
            self.use_cpu = True

        self.infer_time_s = 0
        self.samples = 0

        self.start = 0

        self.device_info = {
            "cpu_rss_mb": None,
            "gpu_rss_mb": None,
            "gpu_util": None
        }
        if mem_info is not None:
            self.mem_info = mem_info

    def tic(self):
        self.start = time.time()

    def toc(self, samples=1):
        self.infer_time_s += (time.time() - self.start)
        self.samples += samples

    def get_device_info(self, cpu_rss_mb=None, gpu_rss_mb=None, gpu_util=None):
        self.device_info["cpu_rss_mb"] = cpu_rss_mb
        self.device_info["gpu_rss_mb"] = gpu_rss_mb
        self.device_info["gpu_util"] = gpu_util

    def report(self):
        model_info = {
            'model_name': self.model_name,
            'precision': self.precision
        }
        data_info = {
            'batch_size': self.batch_size,
            'shape': 'dynamic_shape',
            'data_num': self.samples
        }
        perf_info = {'inference_time_s': self.infer_time_s}
        log = PaddleInferBenchmark(self.config, model_info, data_info,
                                   perf_info, self.device_info)
        log('Test')

    @staticmethod
    def get_current_memory_mb(gpu_id=None):
        pid = os.getpid()
        p = psutil.Process(pid)
        info = p.memory_full_info()
        cpu_rss_mb = info.uss / 1024. / 1024.
        gpu_rss_mb = 0
        if gpu_id is not None:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_rss_mb = meminfo.used / 1024. / 1024.
        return cpu_rss_mb, gpu_rss_mb

    @staticmethod
    def get_current_gputil(gpu_id=None):
        gpu_load = 0
        if gpu_id is not None:
            GPUs = GPUtil.getGPUs()
            gpu_load = GPUs[gpu_id].load
        return gpu_load
