# Copyright 2020-present the HuggingFace Inc. team.
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

import time

import paddle

from paddlenlp.utils.log import logger


class _Timer:
    """Profile Timer for recording time taken by forward/ bacward/ reduce/ step."""

    def __init__(self, name):
        self.name = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, "timer has already started"
        paddle.device.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timers."""
        assert self.started_, "timer is not started."
        paddle.device.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=True):
        """Write timers to a tensorboard writer"""
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar("timers/" + name, value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        # string = "time (ms) / rate"
        string = "time (ms)"

        time_dict = {}
        for name in names:
            time_dict[name] = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer

        # total_time = sum(list(time_dict.values()))
        # string += " | total_time : {:.2f} ".format(total_time)
        time_dict = sorted(time_dict.items(), key=lambda x: x[1], reverse=True)

        for time_tuple in time_dict:
            name, value = time_tuple
            # string += " | {} : {:.2f} ({:.2f}%) ".format(name, value, value * 100.0 / total_time)
            string += " | {} : {:.2f}".format(name, value)
        return string


_GLOBAL_TIMERS = None


def get_timers():
    global _GLOBAL_TIMERS
    return _GLOBAL_TIMERS


def set_timers():
    global _GLOBAL_TIMERS
    logger.info("enable PaddleNLP timer")
    _GLOBAL_TIMERS = Timers()


def disable_timers():
    global _GLOBAL_TIMERS
    logger.info("disable PaddleNLP timer")
    _GLOBAL_TIMERS = None
