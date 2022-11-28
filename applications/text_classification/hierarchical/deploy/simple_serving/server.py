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

from paddlenlp import SimpleServer
from paddlenlp.server import CustomModelHandler, MultiLabelClassificationPostHandler

app = SimpleServer()
app.register('cls_hierarchical',
             model_path="../../export",
             tokenizer_name='ernie-3.0-medium-zh',
             model_handler=CustomModelHandler,
             post_handler=MultiLabelClassificationPostHandler)
