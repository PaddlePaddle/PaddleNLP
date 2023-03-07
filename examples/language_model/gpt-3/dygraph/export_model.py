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

import argparse
import copy
import os
import re

import numpy as np
import paddle

from paddlenlp.transformers import (  # GPTChineseTokenizer,; GPTForGreedyGeneration,
    GPTConfig,
    GPTForGeneration,
    GPTTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPTForGeneration, GPTTokenizer)
    # "gpt2": (GPTLMHeadModel, GPTTokenizer)
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/gpt",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


PREFIX_CHECKPOINT_DIR = "model_state"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\_mp_(\d+)" + ".pdparams$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    if "model_state.pdparams" in content:
        return ["model_state.pdparams"]

    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isfile(os.path.join(folder, path))
    ]
    print("checkpoints", checkpoints)
    if len(checkpoints) == 0:
        raise ValueError("No checkpoint found within folder {}".format(folder))

    return [
        os.path.join(folder, v) for v in sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]))
    ]


def MergedKeys(num_layers):
    res = {}
    Column = [
        "gpt.decoder.layers.0.linear1.bias",
        "gpt.decoder.layers.0.linear1.weight",
        "gpt.decoder.layers.0.self_attn.qkv_proj.bias",
        "gpt.decoder.layers.0.self_attn.qkv_proj.weight",
    ]

    Row = [
        "gpt.embeddings.word_embeddings.weight",
        # 'gpt.decoder.layers.0.self_attn.out_proj.bias',
        "gpt.decoder.layers.0.self_attn.out_proj.weight",
        # 'gpt.decoder.layers.0.linear2.bias',
        "gpt.decoder.layers.0.linear2.weight",
    ]
    for v in Column:
        if "layers.0." in v:
            for i in range(num_layers):
                res[v.replace("layers.0.", f"layers.{i}.")] = "col"
        else:
            res[v] = "col"
    for v in Row:
        if "layers.0." in v:
            for i in range(num_layers):
                res[v.replace("layers.0.", f"layers.{i}.")] = "row"
        else:
            res[v] = "row"

    return res


def merge_rows(values):
    return np.concatenate(values, axis=0)


def merge_column(values):
    return np.concatenate(values, axis=-1)


def merge_model_parallel(model_path, config, as_float32=True):
    weights_path = get_last_checkpoint(model_path)
    if len(weights_path) == 1:
        return paddle.load(weights_path[0], return_numpy=True)

    weights_list = []
    for path in weights_path:
        weights_list.append(paddle.load(path, return_numpy=True))

    final_weight = copy.deepcopy(weights_list[0])
    merged_keys = MergedKeys(config.num_hidden_layers)

    for k, func_name in merged_keys.items():
        func = merge_column if "col" == func_name else merge_rows
        final_weight[k] = func([weight[k] for weight in weights_list])

    if as_float32:
        for k in final_weight.keys():
            final_weight[k] = final_weight[k].astype("float32")

    return final_weight


def left_padding(inputs, pad_id, padding="longest"):
    assert "input_ids" in inputs, "input_ids should be in inputs!"
    max_length = 0
    for ids in inputs["input_ids"]:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(name, max_length, to_pad_id):
        values = inputs[name]
        res = []
        for index, value in enumerate(values):
            res.append(extend_max_lenth(value, max_length, to_pad_id))
        inputs[name] = res

    extend_filed("input_ids", max_length, pad_id)
    if "attention_mask" in inputs:
        extend_filed("attention_mask", max_length, 0)
    if "position_ids" in inputs:
        extend_filed("position_ids", max_length, 0)

    return inputs


def main():
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Suild model and load trained parameters
    tokenizer = tokenizer_class.from_pretrained(args.model_path)
    # model = model_class.from_config(args.model_path, max_predict_len=32, eol_token_id=tokenizer.eol_token_id)
    # config = GPTConfig.from_pretrained(args.model_path)
    # args.model_path = "gpt2-medium-en"
    config = GPTConfig.from_pretrained(args.model_path)

    config.fuse_qkv = True
    # config.max_predict_len = 8
    config.max_dec_len = 20
    config.eos_token_id = tokenizer.eos_token_id
    config.eol_token_id = tokenizer.eol_token_id
    config.pad_token_id = tokenizer.eos_token_id
    config.use_cache = True
    config.top_k = 1

    model = model_class(config)
    # model = model_class.from_pretrained(args.model_path, config=config)
    missing_keys, unexpected_keys = model.set_state_dict(merge_model_parallel(args.model_path, config))
    print("missing_keys", missing_keys)
    print("unexpected_keys", unexpected_keys)
    # Switch to eval model
    model.eval()
    # Convert to static graph with specific input description
    input_text = ["Nice to meet", "Hello "]
    inputs = tokenizer(input_text)

    # input_ids = tokenizer.encode(input_text)['input_ids']
    inputs = tokenizer(input_text)
    inputs = left_padding(inputs, tokenizer.bos_token_id)
    input_ids = inputs["input_ids"]

    input_ids = paddle.to_tensor(input_ids, dtype="int64")
    ret = model(input_ids=input_ids)

    # ret =  model.generate(input_ids = data["input_ids"])
    for x in ret[0].tolist():
        print("==" * 30)
        print(tokenizer.convert_ids_to_string(x))

    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        ],
    )

    # # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
