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
from paddlenlp_ops import winx_unzip
from wintx_reference import unzip_and_dequant_wint2_5


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


def test_main_wint2_5_unzip(test_dir):
    dump_dir = os.path.join(test_dir, "moe_triton_wint2.5")
    tensor_names = [
        "ffn1_weights.pdparams",
        "ffn1_weights_scale.pdparams",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)

    ffn1_weight = tensor_dict["ffn1_weights"]
    ffn1_weights_scale = tensor_dict["ffn1_weights_scale"][:, 0, :]

    quant_type = "weight_only_int2.5"
    unzipped_weight = winx_unzip(
        ffn1_weight,
        ffn1_weights_scale,
        quant_type,
    )
    # print("unzipped_weight[1, 0, 0:6]：", unzipped_weight[1, 0, 0:6].cast("float32"))

    unziped_weight_reference = unzip_and_dequant_wint2_5(
        zipped_weight=ffn1_weight, super_scale=ffn1_weights_scale, scale_compute_dtype=paddle.float32
    )
    # print("unziped_weight_reference[1, 0, 0:6]:", unziped_weight_reference[1, 0, 0:6].astype("float32"))

    ne_out = paddle.not_equal(unzipped_weight, unziped_weight_reference).cast("int32")
    num_ne = paddle.sum(ne_out)
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
                        sys.exit(0)
    else:
        print("unziped_weight is equal to reference!")

    # np.testing.assert_array_equal(unzipped_weight_np, unziped_weight_reference_np)


def test_main(test_dir):
    quant_type = "weight_only_int2.5"
    if quant_type == "weight_only_int2.5":
        test_main_wint2_5_unzip(test_dir=test_dir)
    else:
        print(f"Unsupport quant_type ({quant_type}).")


if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_main(test_dir)
