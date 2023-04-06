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

import unittest

from paddlenlp.metrics import Rouge1, RougeL


class TestRouge(unittest.TestCase):
    def test_rogue1(self):
        rouge1 = Rouge1()
        rouge1.reset()
        cand = ["The", "cat", "The", "cat", "on", "the", "mat"]
        ref_list = [["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]
        self.assertEqual(rouge1.score(cand, ref_list), 0.07692307692307693)

    def test_roguel(self):
        rougel = RougeL()
        rougel.reset()
        cand = ["The", "cat", "The", "cat", "on", "the", "mat"]
        ref_list = [["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]
        rougel.add_inst(cand, ref_list)
        self.assertEqual(rougel.score(), 0.7800511508951408)
