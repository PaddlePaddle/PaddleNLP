# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse

import numpy as np
import paddle
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="inference/model", help="The directory of exported model.")
    return parser.parse_args()


def trans_weight(var):
    shape = var.desc.shape()
    new_shape = [shape[1], shape[0]]
    var.desc.set_shape(new_shape)

    var_data = np.array(var.get_value())
    var.get_value().set(var_data.T, paddle.CPUPlace())


def convert_dequant_scale(var):
    deq_scale = np.array(var.get_value()).astype(np.float32)
    new_deq_scale = np.stack([deq_scale.reshape(-1, 1), np.zeros_like(deq_scale).reshape(-1, 1)], axis=-1).reshape(-1)
    var.get_value().set(np.frombuffer(new_deq_scale.tobytes(), dtype=np.int64), paddle.CPUPlace())


def process_params(model_path):
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    scope = paddle.static.Scope()
    with paddle.base.scope_guard(scope):
        with paddle.base.program_guard(prog, startup_prog):
            [program, feed_target_names, fetch_targets] = paddle.static.io.load_inference_model(model_path, exe)

            feed_targets = []
            for var in program.list_vars():
                if var.name in feed_target_names:
                    feed_targets.append(var)

            block = program.global_block()

            for op in tqdm(block.ops, desc="processing the linear layer for NPU"):
                if op.type == "matmul_v2":
                    w_name = op.input_arg_names[-1]
                    if w_name.endswith("qkv_weight") and not op.attr("trans_y"):
                        op._set_attr("trans_y", True)
                        w = block.var(w_name)
                        trans_weight(w)
                    elif w_name.endswith("out_proj_weight") and not op.attr("trans_y"):
                        op._set_attr("trans_y", True)
                        w = block.var(w_name)
                        trans_weight(w)
                    elif w_name.endswith("ffn1_weight") and not op.attr("trans_y"):
                        op._set_attr("trans_y", True)
                        w = block.var(w_name)
                        trans_weight(w)
                    elif w_name.endswith("ffn2_weight") and not op.attr("trans_y"):
                        op._set_attr("trans_y", True)
                        w = block.var(w_name)
                        trans_weight(w)
                    elif w_name == "llama_lm_head_0.w_0" and not op.attr("trans_y"):
                        op._set_attr("trans_y", True)
                        w = block.var(w_name)
                        trans_weight(w)

            for var_name in tqdm(block.vars, desc="processing the dequant layer for NPU"):
                if var_name.endswith("qkv_out_scale"):
                    var = block.var(var_name)
                    convert_dequant_scale(var)
                elif var_name.endswith("linear_out_scale"):
                    var = block.var(var_name)
                    convert_dequant_scale(var)
                elif var_name.endswith("ffn1_out_scale"):
                    var = block.var(var_name)
                    convert_dequant_scale(var)
                elif var_name.endswith("ffn2_out_scale"):
                    var = block.var(var_name)
                    convert_dequant_scale(var)

            paddle.static.save_inference_model(
                model_path, feed_targets, fetch_targets, exe, program=program, skip_prune_program=True
            )


def main():
    args = parse_arguments()
    process_params(args.model_path)


if __name__ == "__main__":
    main()
