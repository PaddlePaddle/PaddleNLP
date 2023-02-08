# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
from pprint import pprint

import numpy as np
from paddle import inference

from paddlenlp.ops.ext_utils import load
from paddlenlp.transformers import PegasusChineseTokenizer


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_model_dir",
        default="../../inference_model/",
        type=str,
        help="Path to save inference model of Pegasus. ",
    )
    args = parser.parse_args()
    return args


def setup_predictor(args):
    """Setup inference predictor."""
    # Load FastGeneration lib.
    load("FastGeneration", verbose=True)
    model_file = os.path.join(args.inference_model_dir, "pegasus.pdmodel")
    params_file = os.path.join(args.inference_model_dir, "pegasus.pdiparams")
    if not os.path.exists(model_file):
        raise ValueError("not find model file path {}".format(model_file))
    if not os.path.exists(params_file):
        raise ValueError("not find params file path {}".format(params_file))
    config = inference.Config(model_file, params_file)
    config.enable_use_gpu(100, 0)
    config.switch_ir_optim()
    config.enable_memory_optim()
    config.disable_glog_info()

    predictor = inference.create_predictor(config)
    return predictor


def convert_example(example, tokenizer, max_seq_len=512):
    """Convert all examples into necessary features."""
    tokenized_example = tokenizer(
        example, max_length=max_seq_len, padding=True, truncation=True, return_attention_mask=False
    )
    return tokenized_example


def infer(args, predictor):
    """Use predictor to inference."""
    tokenizer = PegasusChineseTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese")

    inputs = [
        "在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！",
        "据微信公众号“界面”报道，4日上午10点左右，中国发改委反垄断调查小组突击查访奔驰上海办事处，调取数据材料，并对多名奔驰高管进行了约谈。截止昨日晚9点，包括北京梅赛德斯-奔驰销售服务有限公司东区总经理在内的多名管理人员仍留在上海办公室内",
    ]

    data = convert_example(inputs, tokenizer, max_seq_len=128)
    input_handles = {}
    for name in predictor.get_input_names():
        input_handles[name] = predictor.get_input_handle(name)
        input_handles[name].copy_from_cpu(np.array(data[name], dtype="int32"))

    output_handles = [predictor.get_output_handle(name) for name in predictor.get_output_names()]

    predictor.run()

    output = [output_handle.copy_to_cpu() for output_handle in output_handles]

    for idx, sample in enumerate(output[0]):
        for beam_idx, beam in enumerate(sample):
            if beam_idx >= len(sample) // 2:
                break
            print(
                f"Example {idx} beam {beam_idx}: ",
                "".join(tokenizer.decode(beam, skip_special_tokens=True, clean_up_tokenization_spaces=False)),
            )


if __name__ == "__main__":
    args = setup_args()
    pprint(args)
    predictor = setup_predictor(args)
    infer(args, predictor)
