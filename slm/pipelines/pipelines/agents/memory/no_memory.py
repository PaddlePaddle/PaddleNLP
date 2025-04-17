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

from typing import Any, Dict, List, Optional

from pipelines.agents.memory import Memory


class NoMemory(Memory):
    """
    A memory class that doesn't store any data.
    """

    def load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
        """
        Load an empty dictionary.

        :param keys: Optional list of keys (ignored in this implementation).
        :return: An empty str.
        """
        return ""

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save method that does nothing.

        :param data: A dictionary containing the data to save (ignored in this implementation).
        """
        pass

    def clear(self) -> None:
        """
        Clear method that does nothing.
        """
        pass
