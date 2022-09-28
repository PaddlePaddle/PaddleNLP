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

import numpy.random
import paddle
from paddlenlp.metrics.glue import MultiLabelsMetric
from sklearn.metrics import precision_recall_fscore_support

import unittest


class TestMultiLabelsMetric(unittest.TestCase):

    def setUp(self):
        self.cls_num = 10
        self.shape = (5, 20, self.cls_num)
        self.label_shape = (5, 20)
        self.metrics = MultiLabelsMetric(num_labels=self.cls_num)

    def get_multi_labels_random_case(self):
        label = np.random.randint(self.cls_num,
                                  size=self.label_shape).astype("int64")
        pred = np.random.uniform(0.1, 1.0,
                                 self.shape).astype(paddle.get_default_dtype())
        np_label = label.reshape(-1)
        np_pred = pred.reshape(-1, self.cls_num).argmax(axis=1)
        average_type = ['micro', 'macro', 'weighted', None]
        pos_label = np.random.randint(0, self.cls_num)
        return label, pred, np_label, np_pred, average_type[np.random.randint(
            0, 3)], pos_label

    def test_compute(self):
        for i in range(29):
            numpy.random.seed(i)
            self.metrics.reset()
            label, pred, np_label, np_pred, average_type, pos_label = self.get_multi_labels_random_case(
            )
            precision, recall, f, _ = precision_recall_fscore_support(
                np_label, np_pred, average=average_type, pos_label=pos_label)
            args = self.metrics.compute(paddle.to_tensor(pred),
                                        paddle.to_tensor(label))
            self.metrics.update(args)
            result = self.metrics.accumulate(average=average_type,
                                             pos_label=pos_label)
            self.assertEqual(precision, result[0])
            self.assertEqual(recall, result[1])
            self.assertEqual(f, result[2])

    def test_reset(self):
        self.metrics.reset()
        numpy.random.seed(0)
        label, pred, np_label, np_pred, average_type, pos_label = self.get_multi_labels_random_case(
        )
        args = self.metrics.compute(paddle.to_tensor(pred),
                                    paddle.to_tensor(label))
        self.metrics.update(args)

        numpy.random.seed(1)
        label, pred, np_label, np_pred, average_type, pos_label = self.get_multi_labels_random_case(
        )
        precision, recall, f, _ = precision_recall_fscore_support(
            np_label, np_pred, average=average_type, pos_label=pos_label)
        args = self.metrics.compute(paddle.to_tensor(pred),
                                    paddle.to_tensor(label))
        self.metrics.update(args)
        result = self.metrics.accumulate(average=average_type,
                                         pos_label=pos_label)
        self.assertNotEqual(precision, result[0])
        self.assertNotEqual(recall, result[1])
        self.assertNotEqual(f, result[2])

        self.metrics.reset()
        args = self.metrics.compute(paddle.to_tensor(pred),
                                    paddle.to_tensor(label))
        self.metrics.update(args)
        result = self.metrics.accumulate(average=average_type,
                                         pos_label=pos_label)
        self.assertEqual(precision, result[0])
        self.assertEqual(recall, result[1])
        self.assertEqual(f, result[2])

    def test_update_accumulate(self):
        steps = 10
        np_pred = np.zeros((0), dtype=int)
        np_label = np.zeros((0), dtype=int)
        for i in range(steps):
            numpy.random.seed(i)
            label, pred, cur_np_label, cur_np_pred, average_type, pos_label = self.get_multi_labels_random_case(
            )
            np_label = np.concatenate((np_label, cur_np_label))
            np_pred = np.concatenate((np_pred, cur_np_pred))
            precision, recall, f, _ = precision_recall_fscore_support(
                np_label, np_pred, average=average_type, pos_label=pos_label)
            args = self.metrics.compute(paddle.to_tensor(pred),
                                        paddle.to_tensor(label))
            self.metrics.update(args)
            result = self.metrics.accumulate(average=average_type,
                                             pos_label=pos_label)
            self.assertEqual(precision, result[0])
            self.assertEqual(recall, result[1])
            self.assertEqual(f, result[2])

    def get_binary_labels_random_case(self):
        label = np.random.randint(self.cls_num,
                                  size=self.label_shape).astype("int64")
        pred = np.random.uniform(0.1, 1.0,
                                 self.shape).astype(paddle.get_default_dtype())
        average_type = 'binary'
        pos_label = np.random.randint(0, self.cls_num)

        np_label = label.reshape(-1)
        selection = pos_label == np_label
        np_label = np.zeros_like(np_label)
        np_label[selection] = 1

        np_pred = pred.reshape(-1, self.cls_num).argmax(axis=1)
        selection = pos_label == np_pred
        np_pred = np.zeros_like(np_pred)
        np_pred[selection] = 1
        return label, pred, np_label, np_pred, average_type, pos_label

    def test_binary_compute(self):
        for i in range(29):
            numpy.random.seed(i)
            self.metrics.reset()
            label, pred, np_label, np_pred, average_type, pos_label = self.get_binary_labels_random_case(
            )
            precision, recall, f, _ = precision_recall_fscore_support(
                np_label, np_pred, average=average_type)
            args = self.metrics.compute(paddle.to_tensor(pred),
                                        paddle.to_tensor(label))
            self.metrics.update(args)
            result = self.metrics.accumulate(average=average_type,
                                             pos_label=pos_label)
            self.assertEqual(precision, result[0])
            self.assertEqual(recall, result[1])
            self.assertEqual(f, result[2])


if __name__ == "__main__":
    unittest.main()
