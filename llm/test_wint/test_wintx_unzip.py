import os

import numpy as np
import paddle
from paddlenlp_ops import winx_unzip



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
    shifts = paddle.to_tensor([13, 11, 9, 6, 4, 2, 0]).unsqueeze(-1).cast('int32')
    ffn_out = winx_unzip(
        ffn1_weight,
        ffn1_weights_scale,
        shifts,
        quant_type,
    )
    print(ffn_out)


def test_main():
    test_dir = "/project/PaddleNLP/llm/test_wint"
    quant_type = "weight_only_int2.5"
    if quant_type == "weight_only_int2.5":
        test_main_wint2_5_unzip(test_dir=test_dir)
    else:
        print(f"Unsupport quant_type ({quant_type}).")


if __name__ == "__main__":
    test_main()
