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
from typing import List, Str, Union

from paddlenlp.trainer import Trainer

from .comm_utils import get_timer_label


@contextmanager
def timers_scope(trainer: Trainer, name, minus_names:Union[List, Str]=None):
    """
    Timing scope that will be used when training.
    Args:
        trainer (Trainer): The trainer object.
        name (str): Name of the timer.
        minus_name (str): Name of the timer to subtract from.
    """
    label = get_timer_label(name)
    trainer.timers and trainer.timers(label).start()
    yield
    trainer.timers and trainer.timers(label).stop()
    if minus_names is not None:
        # if minus_names is a list, then we need to get the label for each name
        if isinstance(minus_names, list):
            minus_labels = [get_timer_label(name) for name in minus_names]
        else:
            minus_labels = [get_timer_label(minus_names)]
        if trainer.timers:
            for minus_label in minus_labels:
                trainer.timers(label).elapsed_ -= trainer.timers(label).elapsed_(minus_label)
    return


@contextmanager
def timers_scope_manual_label(trainer, name, minus_names:Union[List, Str]=None):
    """
    Timing scope that will be used when training.
    Args:
        trainer (Trainer): The trainer object.
        name (str): Name of the timer.
        minus_name (str): Name of the timer to subtract from.
    """
    label = name
    trainer.timers and trainer.timers(label).start()
    yield
    trainer.timers and trainer.timers(label).stop()
    if minus_names is not None:
        # if minus_names is a list, then we need to get the label for each name
        if isinstance(minus_names, list):
            minus_labels = [name for name in minus_names]
        else:
            minus_labels = [minus_names]
        if trainer.timers:
            for minus_label in minus_labels:
                trainer.timers(label).elapsed_ -= trainer.timers(label).elapsed_(minus_label)
    return
