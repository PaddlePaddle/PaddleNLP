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

from contextlib import contextmanager
from typing import List, Union

from paddlenlp.trainer.plugins.timer import RuntimeTimer
from paddlenlp.utils.log import logger

from .comm_utils import get_timer_label


class TimerScope:
    def __init__(self, timers, name: str, minus_names: Union[List[str], str] = None):
        self.timers = timers
        self.name = name
        self.minus_names = minus_names
        if self.minus_names:
            self.minus_labels = [
                self._get_timer_label(name)
                for name in (self.minus_names if isinstance(self.minus_names, list) else [self.minus_names])
            ]
        self.label = self._get_timer_label(name)
        self._started = False  # 跟踪计时器状态

    def start(self) -> None:
        """显式启动计时器"""
        if self.timers:
            self.timers(self.label).start()
            self._started = True

    def stop(self) -> None:
        """显式停止计时器并处理减法逻辑"""
        if self.timers and self._started:
            timer = self.timers(self.label)
            timer.stop()

            if self.minus_names:
                for label in self.minus_labels:
                    timer.elapsed_ -= self.timers(label).elapsed_
            self._started = False

    @staticmethod
    def _get_timer_label(name: str) -> str:
        # 根据实际标签生成逻辑修改
        return get_timer_label(name)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class TimerScopeManualLabel(TimerScope):
    @staticmethod
    def _get_timer_label(name: str) -> str:
        # 根据实际标签生成逻辑修改
        return name


@contextmanager
def timers_scope_runtimer(name):
    """
    Timing scope that will be used when training.
    Args:
        name (str): Name of the timer.
    """
    timer = RuntimeTimer(name)

    timer.start(name)
    yield
    logger.info(f"{timer.log()}")
    return
