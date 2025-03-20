# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

import paddle
from paddle.base import core

_ep_comm_timer = None


def get_ep_timer(buffer=None, enable_timer=True):
    global _ep_comm_timer
    if _ep_comm_timer is None:
        _ep_comm_timer = EPCommTimer()

    if buffer is not None:
        _ep_comm_timer.set_stream(buffer.runtime.get_comm_stream())

    _ep_comm_timer.set_timer(enable_timer)
    return _ep_comm_timer


def all_gather(value):
    value_tensor = paddle.to_tensor([value], dtype='float64')
    global_value_tensor_list = []
    paddle.distributed.all_gather(global_value_tensor_list, value_tensor)
    return global_value_tensor_list


class GPUEventTimers:
    def __init__(self):
        self.timers = {}

    def get(self, name):
        if self.timers.get(name, None) is None:
            device_id = int(paddle.device.get_device().replace("gpu:", ""))
            print(f"-- [GPUEventTimers] device_id: {device_id}")
            self.timers[name] = core.GPUEventTimer(core.CUDAPlace(device_id))
        return self.timers[name] 

    def names(self):
        return self.timers.keys()

    def elapsed_list(self, name, reset):
        return self.get(name).elapsed_list(reset)


class EPCommTimer:
    def __init__(self):
        self._enabled = False
        self._stream = None
        self._timers = []
        self._step_id = 0

    def set_stream(self, stream):
        self._stream = stream

    def set_timer(self, enabled):
        self._enabled = enabled

    def add_step(self):
        self._step_id += 1

    def start(self, name):
        """
        Start the timer, which can capture the execution time on the comm_stream with cudaEvent.
        """
        if not self._enabled:
            return

        assert self._stream is not None
        if len(self._timers) <= self._step_id:
            self._timers.append(GPUEventTimers())

        self._timers[self._step_id].get(name).start(self._stream)

    def stop(self, name):
        """
        Stop the timer, which can capture the execution time on the comm_stream with cudaEvent.
        """
        if not self._enabled:
            return

        assert len(self._timers) > self._step_id
        assert self._stream is not None
        self._timers[self._step_id].get(name).stop(self._stream)

    def sumary(self, reset=True):
        """
        Print execution times for all captured kernels when enable_timer is set.
        """
        if not self._enabled or self._timers is None:
            return 

        string = "time (ms)\n"
        for i in range(len(self._timers)):
            gpu_event_timers = self._timers[i]
            for name in gpu_event_timers.names():
                elapsed_times = np.array(gpu_event_timers.elapsed_list(name, reset)) * 1000.0
                max_time = np.max(elapsed_times)
                min_time = np.min(elapsed_times)
                avg_time = np.mean(elapsed_times)
                total = np.sum(elapsed_times)
                global_max_time = np.max(np.array(all_gather(max_time)))
                global_min_time = np.min(np.array(all_gather(min_time)))
                global_avg_time = np.mean(np.array(all_gather(avg_time)))
                global_total = np.mean(np.array(all_gather(total)))
                string += f"\t step {i} | {name}: total={total:.3f}, max={max_time:.3f}, min={min_time:.3f}, avg={avg_time:.3f}, n={len(elapsed_times)} | global_total={global_total:.3f}, global_max={global_max_time:.3f}, global_min={global_min_time:.3f}, global_avg={global_avg_time:.3f} \n"
            string += "\n"
        print(f"{string}")
