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

import os
import argparse
import numpy as np
from pprint import pprint

import paddle
import paddle.inference as paddle_infer

from paddlenlp.transformers import Ernie3PromptForGeneration, Ernie3PromptTokenizer
from paddlenlp.ops.ext_utils import load


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_model_dir",
        default="./infer_model/",
        type=str,
        help="Path to save inference model of Ernie3Prompt. ")
    parser.add_argument(
        "--model_name_or_path",
        default="ernie3-prompt",
        type=str,
        help="The model name to specify the ernie3 to use. ")
    args = parser.parse_args()

    return args


def infer(args):
    tokenizer = Ernie3PromptTokenizer.from_pretrained(args.model_name_or_path)

    texts = ['生成满足以\"一&乙\"开头的春联:', '生成以\"国赖女娲天再补\"为上联的下联:', '生成包含\"景物\"的春联:']

    encoded_inputs = tokenizer.gen_encode(texts)

    # Load FasterTransformer lib. 
    load("FasterTransformer", verbose=True)

    config = paddle_infer.Config(
        os.path.join(args.inference_model_dir, "ernie3_prompt.pdmodel"),
        os.path.join(args.inference_model_dir, "ernie3_prompt.pdiparams"))
    config.enable_use_gpu(100, 0)
    config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)

    input_handles = {}
    for name in predictor.get_input_names():
        input_handles[name] = predictor.get_input_handle(name)
        if name == "attention_mask":
            input_handles[name].copy_from_cpu(
                np.asarray(
                    encoded_inputs[name], dtype="float32"))
        else:
            input_handles[name].copy_from_cpu(
                np.asarray(
                    encoded_inputs[name], dtype="int32"))

    output_handles = [
        predictor.get_output_handle(name)
        for name in predictor.get_output_names()
    ]

    predictor.run()

    output = [output_handle.copy_to_cpu() for output_handle in output_handles]
    
    if output[0].shape==3:
        # beam search
        for idx, sample in enumerate(output[0].tolist()):
            print(texts[idx])
            for beam_idx, beam in enumerate(sample):
                seq = tokenizer.convert_ids_to_string(beam)
                score = output[1][idx][beam_idx]
                print(f'{idx}-{beam_idx}: {seq}: {score}')
            print('\n')
    else:
        # Sampling
        for idx, sample in enumerate(output[0].tolist()):
            print(texts[idx])
            seq = tokenizer.convert_ids_to_string(sample)
            print(f'{idx}: {seq}')
            print('\n')


if __name__ == "__main__":
    args = setup_args()
    pprint(args)

    infer(args)
