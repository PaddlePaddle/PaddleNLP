# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Memory(ABC):
    """
    Abstract base class for memory management in an Agent.
    """

    @abstractmethod
    def load(self, keys: Optional[List[str]] = None, **kwargs) -> Any:
        """
        Load the context of this model run from memory.

        :param keys: Optional list of keys to specify the data to load.
        :return: The loaded data.
        """

    @abstractmethod
    def save(self, data: Dict[str, Any]) -> None:
        """
        Save the context of this model run to memory.

        :param data: A dictionary containing the data to save.
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Clear memory contents.
        """
