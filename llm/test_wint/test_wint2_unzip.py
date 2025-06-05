# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys

import paddle
from paddlenlp_ops import win2_unzip
from wint2_reference import unzip_and_dequant_wint2


def print_tensor_info(t, name):
    if t is not None:
        print(f"-- [print_tensor_info] {name}: shape={t.shape}, dtype={t.dtype}")
    else:
        print(f"-- [print_tensor_info] {name}: tensor is {t}")


def load_all_tensors(tensor_names, dump_dir):
    tensor_dict = {}
    for name in tensor_names:
        key = name.replace(".pdparams", "").replace("_layer1", "")
        filepath = os.path.join(dump_dir, name)
        tensor_dict[key] = paddle.load(filepath)
        print_tensor_info(tensor_dict[key], name)
    return tensor_dict


def test_main_wint2_unzip(test_dir):
    dump_dir = os.path.join(test_dir, "moe_cluster2_params")
    tensor_names = [
        "w1.pdparams",
        "w1_scale.pdparams",
        "w1_code_scale.pdparams",
        "w1_code_zp.pdparams",
        "w1_super_scale.pdparams",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)

    w = tensor_dict["w1"]
    w_scale = tensor_dict["w1_scale"]
    w_code_scale = tensor_dict["w1_code_scale"]
    w_code_zp = tensor_dict["w1_code_zp"]
    w_super_scale = tensor_dict["w1_super_scale"]

    # tensor_names = [
    #     "w2.pdparams",
    #     "w2_scale.pdparams",
    #     "w2_code_scale.pdparams",
    #     "w2_code_zp.pdparams",
    #     "w2_super_scale.pdparams",
    # ]
    # tensor_dict = load_all_tensors(tensor_names, dump_dir)

    # w = tensor_dict["w2"]
    # w_scale = tensor_dict["w2_scale"]
    # w_code_scale = tensor_dict["w2_code_scale"]
    # w_code_zp = tensor_dict["w2_code_zp"]
    # w_super_scale = tensor_dict["w2_super_scale"]

    quant_type = "weight_only_int2"
    unzipped_weight = win2_unzip(
        w,
        w_scale,
        w_code_scale,
        w_code_zp,
        w_super_scale,
        quant_type,
    )

    unziped_weight_reference = unzip_and_dequant_wint2(
        w=w, w_scale=w_scale, w_code_scale=w_code_scale, w_code_zp=w_code_zp, w_super_scale=w_super_scale
    )
    # print("unziped_weight_reference[1, 0, 0:6]:", unziped_weight_reference[1, 0, 0:6].astype("float32"))

    ne_out = paddle.not_equal(unzipped_weight, unziped_weight_reference).cast("int32")
    num_ne = paddle.sum(ne_out)

    error_num = 0
    if num_ne.item() != 0:
        unzipped_weight_np = unzipped_weight.cast("float32").numpy()
        unziped_weight_reference_np = unziped_weight_reference.cast("float32").numpy()

        w_shape = unzipped_weight.shape
        for i in range(w_shape[0]):
            for j in range(w_shape[1]):
                for k in range(w_shape[2]):
                    if unzipped_weight_np[i, j, k] != unziped_weight_reference_np[i, j, k]:
                        print(
                            f"[{i}, {j}, {k}] mismatch: {unzipped_weight_np[i, j, k]} vs {unziped_weight_reference_np[i, j, k]}"
                        )
                        error_num += 1
                        if error_num > 20:
                            sys.exit(0)
    else:
        print("unziped_weight is equal to reference!")

    # np.testing.assert_array_equal(unzipped_weight_np, unziped_weight_reference_np)


def test_main(test_dir):
    quant_type = "weight_only_int2"
    if quant_type == "weight_only_int2":
        test_main_wint2_unzip(test_dir=test_dir)
    else:
        print(f"Unsupport quant_type ({quant_type}).")


if __name__ == "__main__":
    test_dir = "/paddle/code/data"
    # test_dir = os.path.dirname(os.path.abspath(__file__))
    test_main(test_dir)
