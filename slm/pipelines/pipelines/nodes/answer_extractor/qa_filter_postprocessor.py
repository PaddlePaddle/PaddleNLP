# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
from pipelines.nodes.base import BaseComponent


class QAFilterPostprocessor(BaseComponent):
    """
    QA Filter Postprocessor used to postprocess the result of qa filter.
    """

    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    def __init__(self, device="gpu"):
        paddle.set_device(device)

    def run(self, filtered_cqa_triples):
        results = {
            "documents": [
                {
                    "content": triple["synthetic_question"],
                    "content_type": "text",
                    "meta": {"answer": triple["synthetic_answer"], "_split_id": 0},
                }
                for triple in filtered_cqa_triples
            ]
        }
        return results, "output_1"
