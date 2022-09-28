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
import numpy as np
import unittest
import paddle
import paddlenlp.ops as ops
from common_test import CommonTest

EINSUM_TEST_SAMPLE = {
    "x": np.random.rand(5),
    "y": np.random.rand(7),
    "A": np.random.rand(4, 5),
    "B": np.random.rand(2, 5),
    "C": np.random.rand(3, 7),
    "D": np.random.rand(3, 4, 5),
    "E": np.random.rand(3, 5, 2),
    "F": np.random.rand(2, 4, 5, 3),
    "G": np.random.rand(4, 2, 5),
    "H": np.random.rand(3, 2, 4),
    "I": np.random.rand(2, 2),
    "J": np.random.rand(1, 3, 5),
    "K": np.random.rand(1, 2, 3, 4),
}


class TestEinsum(CommonTest):

    def setUp(self):
        self.sample = {"paradigm": "i->", "data": ["x"]}

    def test_forward(self):
        operands = [
            EINSUM_TEST_SAMPLE[operand] for operand in self.sample["data"]
        ]
        expected_result = np.einsum(self.sample["paradigm"], *operands)

        pd_operands = [paddle.to_tensor(operand) for operand in operands]
        result = ops.einsum(self.sample["paradigm"], *pd_operands)

        self.check_output_equal(result.numpy(), expected_result)

        result2 = ops.einsum(self.sample["paradigm"], pd_operands)
        self.check_output_equal(result2.numpy(), expected_result)


class TestEinsumVectorDot(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "i,i->", "data": ["x", "x"]}


class TestEinsumVectorMul(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "i,i->i", "data": ["x", "x"]}


class TestEinsumVectorOuter(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "i,j->ij", "data": ["x", "y"]}


class TestEinsumMatrixTranspose(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ij->ji", "data": ["A"]}


class TestEinsumMatrixRowSum(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ij->j", "data": ["A"]}


class TestEinsumMatrixColSum(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ij->i", "data": ["A"]}


class TestEinsumMatrixEleMul(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ij,ij->ij", "data": ["A", "A"]}


class TestEinsumMatrixVecMul(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ij,j->i", "data": ["A", "x"]}


class TestEinsumMatrixMul(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ij,kj->ik", "data": ["A", "B"]}


class TestEinsumMatrixOuter(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ij,kl->ijkl", "data": ["A", "C"]}


class TestEinsumTensorBMM(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "bij,bjk->bik", "data": ["D", "E"]}


class TestEinsumTensorContract1(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->i", "data": ["D", "A"]}


class TestEinsumTensorContract2(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ijk,lk->ijl", "data": ["D", "B"]}


class TestEinsumTensorContract3(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "abcd,dfg->abcfg", "data": ["F", "D"]}


class TestEinsumTensorContract4(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->ik", "data": ["D", "A"]}


class TestEinsumTensorContract5(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->ij", "data": ["D", "A"]}


class TestEinsumTensorContract6(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ik, ijk->j", "data": ["A", "G"]}


class TestEinsumTensorContract7(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ijk, ik->jk", "data": ["G", "A"]}


class TestEinsumEllipsis1(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "i...->...", "data": ["G"]}


class TestEinsumEllipsis2(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ij,...i->j...", "data": ["A", "H"]}


class TestEinsumEllipsis3(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "k...,jk", "data": ["F", "I"]}


class TestEinsumTestEinsumBilinear(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "bn,anm,bm->ba", "data": ["B", "E", "I"]}


class TestEinsumTestEinsumOthers(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ijkl, lmn->kmn", "data": ["F", "H"]}


class TestEinsumTestEinsumOthers(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "ijkl, lmn->ijn", "data": ["F", "H"]}


class TestEinsumBatch1(TestEinsum):

    def setUp(self):
        self.sample = {"paradigm": "blq,bhlk->bhlqk", "data": ["J", "K"]}


if __name__ == "__main__":
    unittest.main()
