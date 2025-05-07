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
from typing import List, Optional, Union

from ...trainer.plugins.timer import RuntimeTimer
from ...utils.log import logger
from .comm_utils import get_timer_label


class TimerScope:
    """
    A context manager that provides a timer scope for timing events.
    """

    def __init__(self, timers, name: str, minus_names: Optional[Union[List[str], str]] = None):
        """
        Initialize the TimerScope.

        Args:
            timers (Callable): A function that returns a timer object based on a given label.
            name (str): The name of the timer scope.
            minus_names (Union[List[str], str], optional): A list of timer names or a single timer name to subtract
                their elapsed time from the current timer. Defaults to None.
        """
        self.timers = timers
        self.name = name
        self.minus_names = minus_names
        if self.minus_names:
            self.minus_labels = [
                self._get_timer_label(name)
                for name in (self.minus_names if isinstance(self.minus_names, list) else [self.minus_names])
            ]
        self.label = self._get_timer_label(name)
        self._started = False  # Track the timer status

    def start(self) -> None:
        """
        Explicitly start the timer.
        """
        if self.timers:
            self.timers(self.label).start()
            self._started = True

    def stop(self) -> None:
        """
        Explicitly stop the timer and handle subtraction logic.
        """
        if self.timers and self._started:
            timer = self.timers(self.label)
            timer.stop()

            if self.minus_names:
                for label in self.minus_labels:
                    timer.elapsed_ -= self.timers(label).elapsed_
            self._started = False

    @staticmethod
    def _get_timer_label(name: str) -> str:
        """
        Get the timer label and apply logical modifications.

        Args:
            name (str): The name of the timer.

        Returns:
            str: The modified timer label after applying logical changes.
        """
        # Modify the label based on actual logic
        return get_timer_label(name)

    def __enter__(self):
        """
        Start the timer and return itself.
        This method is called when using the with statement for context management.

        Returns:
            TimeCounter: Returns itself to allow subsequent operations.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the timer and release resources.
        This method is automatically called when using the with statement.

        Args:
            exc_type (Optional[Type[BaseException]]): Optional, the type of the exception, defaults to None.
                If not None, it indicates an exception occurred during execution, and this parameter will be the type of that exception.
            exc_val (Optional[BaseException]): Optional, the value of the exception, defaults to None.
                If not None, it indicates an exception occurred during execution, and this parameter will be the instance of that exception.
            exc_tb (Optional[TracebackType]): Optional, the traceback information, defaults to None.
                If not None, it indicates an exception occurred during execution, and this parameter will be a Traceback object containing the traceback information.

        Returns:
            None: No return value.
        """
        self.stop()


class TimerScopeManualLabel(TimerScope):
    """
    TimerScopeManualLabel is a subclass of TimerScope that overrides the _get_timer_label method.
    It is specifically designed for testing and debugging purposes.
    """

    @staticmethod
    def _get_timer_label(name: str) -> str:
        """
        Generate a logically distinct label based on the given name.
        This function is primarily used for testing and debugging purposes, and can be modified as needed.

        Args:
            name (str): The input name.

        Returns:
            str: A logically distinct name.
        """
        # Apply logical modifications to the label based on actual requirements
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
