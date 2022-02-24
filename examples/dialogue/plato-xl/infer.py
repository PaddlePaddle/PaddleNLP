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
from collections import namedtuple

from utils.args import parse_args, str2bool
from readers.plato_reader import PlatoReader
from readers.dialog_reader import DialogReader

from paddlenlp.transformers import UnifiedTransformerLMHeadModel


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Model")
    group.add_argument("--init_from_ckpt", type=str, default="")
    group.add_argument("--vocab_size", type=int, default=8001)
    group.add_argument("--latent_type_size", type=int, default=20)
    group.add_argument("--num_layers", type=int, default=24)

    group = parser.add_argument_group("Task")
    group.add_argument("--is_cn", type=str2bool, default=False)

    args, _ = parser.parse_known_args()
    DialogReader.add_cmdline_args(parser)

    args = parse_args(parser)
    args.batch_size *= args.latent_type_size

    return args


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


def infer(args):
    model_name = 'plato-xl'
    model = UnifiedTransformerLMHeadModel.from_pretrained(model_name)
    model.eval()

    plato_reader = PlatoReader(args)

    context = [
        "Hi , Becky , what's up ?",
        "Not much , except that my mother-in-law is driving me up the wall .",
        "What's the problem ?"
    ]

    Example = namedtuple("Example", ["src", "data_id"])
    example = Example(src=" [SEP] ".join(context), data_id=0)
    record = plato_reader._convert_example_to_record(example)

    data = plato_reader._pad_batch_records([record], dtype="float16")

    for i in range(200):
        if 100 == i:
            paddle.device.cuda.synchronize()
            start = time.time()

    outputs, _ = model.generate(
        input_ids=data['input_ids'],
        token_type_ids=data['type_ids'],
        position_ids=data['pos_ids'],
        decoder_position_ids=data['decoder_position_ids'],
        attention_mask=data['attention_mask'],
        decoder_type_ids=data['decoder_type_ids'],
        role_ids=data['role_ids'],
        decoder_role_ids=data['decoder_role_ids'],
        seq_len=data['seq_len'],
        max_length=64,
        min_length=1,
        decode_strategy='sampling',
        top_k=5,
        use_fp16_decoding=True,
        use_faster=True)

    paddle.device.cuda.synchronize()
    print("Average time for FasterGeneration is {}ms. ".format((time.time(
    ) - start) / 100 * 1000))

    result = postprocess_response(outputs[0].numpy(), tokenizer)
    result = "".join(result)

    print("Model input:", context)
    print("Result:", result)


if __name__ == "__main__":
    args = setup_args()
    infer(args)
