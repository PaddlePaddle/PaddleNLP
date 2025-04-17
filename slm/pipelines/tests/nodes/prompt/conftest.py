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

from unittest.mock import Mock


def create_mock_layer_that_supports(model_name, response=["fake_response"]):
    """
    Create a mock invocation layer that supports the model_name and returns response.
    """

    def mock_supports(model_name_or_path, **kwargs):
        return model_name_or_path == model_name

    return Mock(**{"model_name_or_path": model_name, "supports": mock_supports, "invoke.return_value": response})
