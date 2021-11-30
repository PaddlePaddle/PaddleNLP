# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.utils.tools import static_params_to_dygraph
import paddle
from paddlenlp.transformers import ErnieModel, ErnieForPretraining, ErniePretrainingCriterion, ErnieTokenizer
import os
import paddle
import paddle.static as static
import paddle.nn as nn


def load_ernie_model(static_model_path):

    model=ErnieModel.from_pretrained('ernie-1.0')
    program_state = static.load_program_state(static_model_path)
    ret_dict=static_params_to_dygraph(model,program_state)

    print('转换前的参数：')
    print(model.embeddings.word_embeddings.weight )
    model.load_dict(ret_dict)
    print('转换后的参数：')
    print(model.embeddings.word_embeddings.weight )
    model.save_pretrained("./ernie_checkpoint")
    


if __name__ == "__main__":
    static_model_path="./output/ernie-1.0-dp8-gb1024/model_last/static_vars"
    load_ernie_model(static_model_path)