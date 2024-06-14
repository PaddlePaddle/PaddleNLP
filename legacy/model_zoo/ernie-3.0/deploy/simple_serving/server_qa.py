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
import numpy as np

from paddlenlp import SimpleServer
from paddlenlp.server import BasePostHandler, QAModelHandler


class QAPostHandler(BasePostHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, data, parameters):
        start_logits = data["logits"]
        end_logits = data["logits_1"]
        contexts = data["data"]["context"]
        questions = data["data"]["question"]
        offset_mappings = data["data"]["offset_mapping"]
        answers = []
        count = 0
        for start_logit, end_logit, offset_mapping in zip(start_logits, end_logits, offset_mappings):
            start_position = np.argmax(np.array(start_logit))
            end_position = np.argmax(np.array(end_logit))
            start_id = offset_mapping[start_position][0]
            end_id = offset_mapping[end_position][1]
            answer = []
            if end_position > start_position:
                answer = contexts[count][start_id:end_id]
            answers.append(answer)
            count += 1

        return {"context": contexts, "question": questions, "answer": answers}


app = SimpleServer()
app.register(
    "models/ernie_qa",
    model_path="../../best_models/cmrc2018/export/",
    tokenizer_name="ernie-3.0-medium-zh",
    model_handler=QAModelHandler,
    post_handler=QAPostHandler,
)
