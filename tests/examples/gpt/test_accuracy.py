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

import unittest
import numpy as np
import os
import paddlenlp


def check_dataset():
    return True


def check_init_checkpoint():
    return True


def get_groundtruth():
    res = {
        1: {
            "loss": 11.008564949
        },
        20: {
            "loss": 10.876321793
        },
    }
    return res


def get_gpt_path():
    return "../../../examples/language_model/gpt"


def get_scripts_path():
    return "../../../tests/examples/gpt"


def parse_log(path=None):
    path = os.path.join(get_gpt_path(), "tests", path)
    if not os.path.exists(path):
        raise ValueError("File not found in %s ." % path)

    os.system('cat %s | grep "global step"  > tmp.log' % path)
    res = {}
    with open(os.path.join(".", "tmp.log"), "r") as f:
        line = f.readline()
        while line:
            line = line.split(" - ")[-1]
            kw = line.split(",")
            global_step = int(kw[0].split(" ")[-1])
            loss = float(kw[3].split(" ")[-1])
            lr = float(kw[6].split(" ")[-1].split("\x1b")[0])

            res[global_step] = {"loss": loss, "lr": lr}

            line = f.readline()
    os.system("rm tmp.log")
    return res


def print_test_results(name):
    print("\n" * 5)
    print("---- This is test reports for %s task: ----" % name)


class GPTAccuarcy(unittest.TestCase):
    """
    Train accuarcy test for GPT
    """

    def test_acc_single_card(self):
        check_dataset()
        check_init_checkpoint()

        for task_name in ["acc_single_dygraph", "acc_single_static"]:
            ret = os.system("cd %s && sh %s/%s.sh" %
                            (get_gpt_path(), get_scripts_path(), task_name))
            if ret != 0:
                print(ret)
                raise ValueError("Train script failed")
            gt = get_groundtruth()
            res = parse_log("./output/gpt-%s/log/workerlog.0" %
                            task_name.replace("_", "-"))
            print_test_results(task_name)
            for k in gt.keys():
                print("%s step: %d, gt:%.9f res:%.9f " %
                      (task_name, k, gt[k]["loss"], res[k]["loss"]))
                self.assertAlmostEqual(gt[k]["loss"],
                                       res[k]["loss"],
                                       delta=1e-6)
            print("\n" * 5)

    def test_acc_dp(self):
        check_dataset()
        check_init_checkpoint()

        for task_name in ["acc_dp_dygraph", "acc_dp_static"]:
            ret = os.system("cd %s && sh %s/%s.sh" %
                            (get_gpt_path(), get_scripts_path(), task_name))
            if ret != 0:
                print(ret)
                raise ValueError("Train script failed")

            gt = get_groundtruth()
            res1 = parse_log("./output/gpt-%s/log/workerlog.0" %
                             task_name.replace("_", "-"))
            res2 = parse_log("./output/gpt-%s/log/workerlog.1" %
                             task_name.replace("_", "-"))

            print_test_results(task_name)
            for k in gt.keys():
                mean = (res1[k]["loss"] + res2[k]["loss"]) / 2
                print("%s step: %d, gt:%.9f res:%.9f " %
                      (task_name, k, gt[k]["loss"], mean))
                self.assertAlmostEqual(gt[k]["loss"], mean, delta=5e-6)
            print("\n" * 5)

    def test_acc_sharding_static(self):
        check_dataset()
        check_init_checkpoint()

        for task_name in ["acc_sharding_static"]:
            ret = os.system("cd %s && sh %s/%s.sh" %
                            (get_gpt_path(), get_scripts_path(), task_name))
            if ret != 0:
                print(ret)
                raise ValueError("Train script failed")

            gt = get_groundtruth()
            res1 = parse_log("./output/gpt-%s/log/workerlog.0" %
                             task_name.replace("_", "-"))
            res2 = parse_log("./output/gpt-%s/log/workerlog.1" %
                             task_name.replace("_", "-"))

            print_test_results(task_name)
            for k in gt.keys():
                mean = (res1[k]["loss"] + res2[k]["loss"]) / 2
                print("%s step: %d, gt:%.9f res:%.9f " %
                      (task_name, k, gt[k]["loss"], mean))
                self.assertAlmostEqual(gt[k]["loss"], mean, delta=5e-6)
            print("\n" * 5)

    @unittest.skipIf(True,
                     "This folder not support MP. Please use MP in GPT-3.")
    def test_acc_mp_static(self):
        check_dataset()
        check_init_checkpoint()

        for task_name in ["acc_mp_static"]:
            ret = os.system("cd %s && sh %s/%s.sh" %
                            (get_gpt_path(), get_gpt_path(), task_name))
            if ret != 0:
                print(ret)
                raise ValueError("Train script failed")

            gt = get_groundtruth()
            res1 = parse_log("./output/gpt-%s/log/workerlog.0" %
                             task_name.replace("_", "-"))
            res2 = parse_log("./output/gpt-%s/log/workerlog.1" %
                             task_name.replace("_", "-"))

            print_test_results(task_name)
            for k in gt.keys():
                self.assertAlmostEqual(res1[k]["loss"],
                                       res2[k]["loss"],
                                       delta=1e-7)
                mean = (res1[k]["loss"] + res2[k]["loss"]) / 2
                print("%s step: %d, gt:%.9f res:%.9f " %
                      (task_name, k, gt[k]["loss"], mean))
                self.assertAlmostEqual(gt[k]["loss"],
                                       res1[k]["loss"],
                                       delta=1e-7)
            print("\n" * 5)


if __name__ == "__main__":
    unittest.main()
