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
import numpy as np
from pprint import pprint

import paddle
import paddle.inference as paddle_infer

from paddlenlp.transformers import MBartTokenizer
from paddlenlp.ops.ext_utils import load


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_model_dir", default="./infer_model/", type=str, help="Path to save inference model of BART. "
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size. ")

    args = parser.parse_args()

    return args


def postprocess_response(tokenizer, seq, bos_idx, eos_idx):
    """Post-process the decoded sequence."""
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [idx for idx in seq[: eos_pos + 1] if idx != bos_idx and idx != eos_idx]
    res = tokenizer.convert_ids_to_string(seq)
    return res


def infer(args):
    model_name = "mbart-large-50-many-to-many-mmt"

    tokenizer = MBartTokenizer.from_pretrained(model_name, src_lang="en_XX")
    bos_id = tokenizer.lang_code_to_id["zh_CN"]
    inputs = "PaddleNLP is a powerful NLP library with Awesome pre-trained models and easy-to-use interface, supporting wide-range of NLP tasks from research to industrial applications."

    eos_id = tokenizer.eos_token_id

    # Input ids
    input_ids = tokenizer(inputs)["input_ids"]
    input_ids = np.asarray(input_ids, dtype="int32").reshape(1, -1).repeat(args.batch_size, axis=0)

    # Forced bos token ids
    forced_bos_token = np.ones([args.batch_size, 1], dtype="int32") * bos_id

    # Load FasterTransformer lib.
    load("FasterTransformer", verbose=True)

    config = paddle_infer.Config(
        os.path.join(args.inference_model_dir, "mbart.pdmodel"),
        os.path.join(args.inference_model_dir, "mbart.pdiparams"),
    )

    config.enable_use_gpu(100, 0)
    config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)

    input_names = predictor.get_input_names()

    # Input ids
    input_ids_handle = predictor.get_input_handle(input_names[0])
    input_ids_handle.copy_from_cpu(input_ids.astype("int32"))

    # Forced bos token ids
    forced_bos_token_handle = predictor.get_input_handle(input_names[1])
    forced_bos_token_handle.copy_from_cpu(forced_bos_token.astype("int32"))

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()

    # [batch_size, num_beams * 2, sequence_length]
    output_data = output_data.transpose([1, 2, 0])

    # Only use the best sequence.
    result = [postprocess_response(tokenizer, sample.tolist()[0], bos_id, eos_id) for sample in output_data]
    print("Model input:", inputs)
    print("Result:", "\n".join(result))


if __name__ == "__main__":
    args = setup_args()
    pprint(args)

    infer(args)
