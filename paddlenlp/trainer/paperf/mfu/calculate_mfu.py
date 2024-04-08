# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

import json
import device
import argparse


class MFUCalculator(object):
    """
    Calculate the MFU for LLM use the method in "Reducing Activation Recomputation in Large Transformer Models".
    """

    def __init__(self, json_path, model_name):
        self.init_from_json(json_path, model_name)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size / self.num_attention_heads

    def init_from_json(self, json_path, model_name):
        configs = {}
        with open(json_path, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                configs[data[i]["model_name"]] = data[i]

        # print(f"<<<<<<<< All Configs: >>>>>>>>")
        # print(configs)

        assert (
            configs.get(model_name, None) is not None
        ), "model_name {model_name} is not implemented!"

        for key, value in configs[model_name].items():
            setattr(self, key, value)
        return self

    def calc_mfu_hfu(self, speed, peak_flops_per_device, recompute_granularity=None):
        """
        Args:
          speed: tokens per second per GPU
          peak_flops_per_device: peek flops of one device
        """
        attn_flop, attn_params = self.calc_attention_flops()
        mlp_flop, mlp_params = self.calc_mlp_flops()
        head_flop, head_params = self.calc_head_flops()

        model_flop_per_batch = (
            (attn_flop + mlp_flop) * self.num_hidden_layers + head_flop
        ) * 3
        if recompute_granularity is None:
            hardware_flop_per_batch = model_flop_per_batch
        elif recompute_granularity == "full":
            hardware_flop_per_batch = (
                model_flop_per_batch + (attn_flop + mlp_flop) * self.num_hidden_layers
            )
        elif recompute_granularity == "full_attn":
            hardware_flop_per_batch = (
                model_flop_per_batch + attn_flop * self.num_hidden_layers
            )
        else:
            assert False

        num_params = (attn_params + mlp_params) * self.num_hidden_layers + head_params

        # (gbs * max_seqlen) / step_time = speed * num_gpus
        # => step_time = (gbs * max_seqlen) / (speed * num_gpus)
        #
        # model_flops = (model_flop_per_batch * gbs) / step_time
        #             = (model_flop_per_batch * gbs) / ((gbs * max_seqlen) / (speed * num_gpus))
        #             = (model_flop_per_batch * speed * num_gpus) / max_seqlen
        model_flops_per_device = (model_flop_per_batch * speed) / (
            self.max_seqlen * 1e12
        )

        # MFU = model_flops_per_device / peak_flops_per_device
        #     = (model_flop_per_batch * speed) / (max_seqlen * peak_flops_per_device)
        mfu = (model_flop_per_batch * speed) / (
            self.max_seqlen * peak_flops_per_device * 1e12
        )
        mfu_coeff = model_flop_per_batch / (
            self.max_seqlen * peak_flops_per_device * 1e12
        )

        hfu = (hardware_flop_per_batch * speed) / (
            self.max_seqlen * peak_flops_per_device * 1e12
        )
        hfu_coeff = hardware_flop_per_batch / (
            self.max_seqlen * peak_flops_per_device * 1e12
        )

        num_params_format = format(num_params, ",")
        print(
            f"- Model Name: {self.model_name}; Parameters: {num_params_format}; Model FLOPs: {model_flops_per_device:.3f}; MFU: {mfu * 100:.2f}% (coeff: {mfu_coeff}); HFU: {hfu * 100:.2f}% (hfu_coeff: {hfu_coeff})"
        )

    def calc_attention_flops(self):
        qkv_trans_flop, qkv_trans_params = self.qkv_trans()
        attn_matrix_flop, attn_matrix_params = self.attn_matrix()
        attn_over_values_flop, attn_over_values_params = self.attn_over_values()
        post_linear_proj_flop, post_linear_proj_params = self.post_linear_proj()

        num_flop_per_batch = (
            qkv_trans_flop * 3
            + attn_matrix_flop
            + attn_over_values_flop
            + post_linear_proj_flop
        )
        num_params = (
            qkv_trans_params * 3
            + attn_matrix_params
            + attn_over_values_params
            + post_linear_proj_params
        )
        return num_flop_per_batch, num_params

    def calc_mlp_flops(self):
        (
            gate_up_linear_proj_flop,
            gate_up_linear_proj_params,
        ) = self.gate_up_linear_proj()
        down_linear_proj_flop, down_linear_proj_params = self.down_linear_proj()
        if self.has_gate_linear:
            num_flop_per_batch = gate_up_linear_proj_flop * 2 + down_linear_proj_flop
            num_params = gate_up_linear_proj_params * 2 + down_linear_proj_params
        else:
            num_flop_per_batch = gate_up_linear_proj_flop + down_linear_proj_flop
            num_params = gate_up_linear_proj_params + down_linear_proj_params
        return num_flop_per_batch, num_params

    def calc_head_flops(self):
        # [bs, seqlen, hidden_size] * [hidden_size, vocab_size] -> [bs, seqlen, vocab_size]
        num_flop_per_batch = self.matmul(
            m=self.max_seqlen, n=self.vocab_size, k=self.hidden_size
        )
        num_params = self.hidden_size * self.vocab_size
        return num_flop_per_batch, num_params

    def qkv_trans(self):
        # [bs, seqlen, hidden_size] * [hidden_size, hidden_size] -> [bs, seqlen, hidden_size]
        num_flop_per_batch = self.matmul(
            m=self.max_seqlen, n=self.hidden_size, k=self.hidden_size
        )
        num_params = self.hidden_size * self.hidden_size
        return num_flop_per_batch, num_params

    def attn_matrix(self):
        # [bs, num_heads, seqlen, head_dim] * [bs, num_heads, seqlen, head_dim] -> [bs, num_heads, seqlen, seqlen]
        num_flop_per_batch = self.batched_matmul(
            bs=self.num_attention_heads,
            m=self.max_seqlen,
            n=self.max_seqlen,
            k=self.head_dim,
        )
        return num_flop_per_batch, 0

    def attn_over_values(self):
        # [bs, num_heads, seqlen, seqlen] * [bs, num_heads, seqlen, head_dim] -> [bs, num_heads, seqlen, head_dim]
        num_flop_per_batch = self.batched_matmul(
            bs=self.num_attention_heads,
            m=self.max_seqlen,
            n=self.head_dim,
            k=self.max_seqlen,
        )
        return num_flop_per_batch, 0

    def post_linear_proj(self):
        # [bs, seqlen, hidden_size] * [hidden_size, hidden_size] -> [bs, seqlen, hidden_size]
        num_flop_per_batch = self.matmul(
            m=self.max_seqlen, n=self.hidden_size, k=self.hidden_size
        )
        num_params = self.hidden_size * self.hidden_size
        return num_flop_per_batch, num_params

    def gate_up_linear_proj(self):
        # [bs, seqlen, hidden_size] * [hidden_size, intermediate_size] -> [bs, seqlen, intermediate_size]
        num_flop_per_batch = self.matmul(
            m=self.max_seqlen, n=self.intermediate_size, k=self.hidden_size
        )
        num_params = self.hidden_size * self.intermediate_size
        return num_flop_per_batch, num_params

    def down_linear_proj(self):
        # [bs, seqlen, intermediate_size] * [intermediate_size, hidden_size] -> [bs, seqlen, hidden_size]
        num_flop_per_batch = self.matmul(
            m=self.max_seqlen, n=self.hidden_size, k=self.intermediate_size
        )
        num_params = self.intermediate_size * self.hidden_size
        return num_flop_per_batch, num_params

    def matmul(self, m, n, k):
        return 2 * m * n * k

    def batched_matmul(self, bs, m, n, k):
        return 2 * bs * m * n * k


def main(args):
    def get_value(args, name):
        if isinstance(args, dict):
            return args.get(name, None)
        else:
            if hasattr(args, name):
                return getattr(args, name)
            return None

    model_json_path = get_value(args, "model_json_path")
    model_name = get_value(args, "model_name")
    calculator = MFUCalculator(model_json_path, model_name)

    if get_value(args, "speed_per_device") is not None:
        speed = get_value(args, "speed_per_device")
    elif (
        get_value(args, "total_speed") is not None
        and get_value(args, "num_gpus") is not None
    ):
        speed = get_value(args, "total_speed") / get_value(args, "num_gpus")
    else:
        assert False, f"Please set speed_per_device or total_speed with num_gpus."

    dev = device.AmpereConfig(device=get_value(args, "device_type"))
    precision = get_value(args, "precision")
    if precision in ["tf32", "fp16", "bf16"]:
        peak_flops_per_device = dev.peak_flops[precision + "_tensor_core"]
    else:
        peak_flops_per_device = dev.peak_flops[precision]

    recompute_granularity = get_value(args, "recompute_granularity")
    calculator.calc_mfu_hfu(speed, peak_flops_per_device, recompute_granularity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MFU Calculator for Transformer Models"
    )
    parser.add_argument(
        "--runtime_json_path", type=str, default=None, help="runtime json path."
    )
    parser.add_argument(
        "--runtime_json_id",
        type=int,
        default=None,
        help="The id in runtime json config.",
    )
    parser.add_argument(
        "--model_json_path", type=str, default=None, help="model json path."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="model name such as facebook/llama-13b.",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        default=None,
        help="Only support 40G-A100, 80G-A100, 80G-A800.",
    )
    parser.add_argument("--precision", type=str, default=None, help="bf16, fp16.")
    parser.add_argument(
        "--speed_per_device",
        type=float,
        default=None,
        help="The speed in tokens/sec/device.",
    )
    parser.add_argument(
        "--total_speed", type=float, default=None, help="The speed in tokens/sec."
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None, help="The number of GPUs."
    )
    parser.add_argument(
        "--recompute_granularity",
        type=str,
        default=None,
        help="Only support full, full_attn.",
    )
    args = parser.parse_args()

    if args.runtime_json_path is not None:
        runtime_args = []
        with open(args.runtime_json_path, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                runtime_args.append(data[i])
        for i in range(len(runtime_args)):
            main(runtime_args[i])
    else:
        main(args)
