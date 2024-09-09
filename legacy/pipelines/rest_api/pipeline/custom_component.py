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
"""
Pipelines allow putting together Components to build a graph.

In addition to the standard pipelines Components, custom user-defined Components
can be used in a Pipeline YAML configuration.

The classes for the Custom Components must be defined in this file.
"""

from pipelines.nodes.base import BaseComponent


class SampleComponent(BaseComponent):
    def run(self, **kwargs):
        raise NotImplementedError
