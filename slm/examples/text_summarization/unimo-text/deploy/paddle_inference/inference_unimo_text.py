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

from paddlenlp.data import Pad
from paddlenlp.ops.ext_utils import load
from paddlenlp.transformers import UNIMOTokenizer


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_model_dir",
        default="../../inference_model",
        type=str,
        help="Path to save inference model of UNIMOText. ",
    )
    args = parser.parse_args()
    return args


def setup_predictor(args):
    """Setup inference predictor."""
    # Load FastGeneration lib.
    load("FastGeneration", verbose=True)
    model_file = os.path.join(args.inference_model_dir, "unimo_text.pdmodel")
    params_file = os.path.join(args.inference_model_dir, "unimo_text.pdiparams")
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


def convert_example(example, tokenizer, max_seq_len=512, return_length=True):
    """Convert all examples into necessary features."""
    source = example
    tokenized_example = tokenizer.gen_encode(
        source,
        max_seq_len=max_seq_len,
        add_start_token_for_decoding=True,
        return_length=True,
        is_split_into_words=False,
    )
    return tokenized_example


def batchify_fn(batch_examples, pad_val, pad_right=False):
    """Batchify a batch of examples."""

    def pad_mask(batch_attention_mask):
        """Pad attention_mask."""
        batch_size = len(batch_attention_mask)
        max_len = max(map(len, batch_attention_mask))
        attention_mask = np.ones((batch_size, max_len, max_len), dtype="float32") * -1e9
        for i, mask_data in enumerate(attention_mask):
            seq_len = len(batch_attention_mask[i])
            if pad_right:
                mask_data[:seq_len:, :seq_len] = np.array(batch_attention_mask[i], dtype="float32")
            else:
                mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i], dtype="float32")
        # In order to ensure the correct broadcasting mechanism, expand one
        # dimension to the second dimension (n_head of Transformer).
        attention_mask = np.expand_dims(attention_mask, axis=1)
        return attention_mask

    pad_func = Pad(pad_val=pad_val, pad_right=pad_right, dtype="int32")
    input_ids = pad_func([example["input_ids"] for example in batch_examples])
    token_type_ids = pad_func([example["token_type_ids"] for example in batch_examples])
    attention_mask = pad_mask([example["attention_mask"] for example in batch_examples])
    seq_len = np.asarray([example["seq_len"] for example in batch_examples], dtype="int32")
    input_dict = {}
    input_dict["input_ids"] = input_ids
    input_dict["token_type_ids"] = token_type_ids
    input_dict["attention_mask"] = attention_mask
    input_dict["seq_len"] = seq_len
    return input_dict


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


def infer(args, predictor):
    """Use predictor to inference."""
    tokenizer = UNIMOTokenizer.from_pretrained("unimo-text-1.0-summary")

    inputs = [
        "雪后的景色可真美丽呀！不管是大树上，屋顶上，还是菜地上，都穿上了一件精美的、洁白的羽绒服。放眼望去，整个世界变成了银装素裹似的，世界就像是粉妆玉砌的一样。",
        "根据“十个工作日”原则，下轮调价窗口为8月23日24时。卓创资讯分析，原油价格或延续震荡偏弱走势，且新周期的原油变化率仍将负值开局，消息面对国内成品油市场并无提振。受此影响，预计国内成品油批发价格或整体呈现稳中下滑走势，但“金九银十”即将到来，卖方看好后期市场，预计跌幅较为有限。",
    ]

    examples = [convert_example(i, tokenizer) for i in inputs]
    data = batchify_fn(examples, tokenizer.pad_token_id)

    input_handles = {}
    for name in predictor.get_input_names():
        input_handles[name] = predictor.get_input_handle(name)
        input_handles[name].copy_from_cpu(data[name])

    output_handles = [predictor.get_output_handle(name) for name in predictor.get_output_names()]

    predictor.run()

    output = [output_handle.copy_to_cpu() for output_handle in output_handles]

    for idx, sample in enumerate(output[0]):
        for beam_idx, beam in enumerate(sample):
            if beam_idx > len(sample) // 2:
                break
            print(f"Example {idx} beam beam_idx {beam_idx}: ", "".join(postprocess_response(beam, tokenizer)))


if __name__ == "__main__":
    args = setup_args()
    pprint(args)
    predictor = setup_predictor(args)
    infer(args, predictor)
