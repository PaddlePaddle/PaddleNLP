# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from seq2seq_attn import Seq2SeqAttnInferModel
from args import parse_args
from data import create_infer_loader


def main():
    args = parse_args()
    _, src_vocab_size, tgt_vocab_size, bos_id, eos_id = create_infer_loader(
        args)

    # Build model and load trained parameters
    model = Seq2SeqAttnInferModel(src_vocab_size,
                                  tgt_vocab_size,
                                  args.hidden_size,
                                  args.hidden_size,
                                  args.num_layers,
                                  args.dropout,
                                  bos_id=bos_id,
                                  eos_id=eos_id,
                                  beam_size=args.beam_size,
                                  max_out_len=256)

    # Load the trained model
    model.set_state_dict(paddle.load(args.init_from_ckpt))

    # Wwitch to eval model
    model.eval()
    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # src
            paddle.static.InputSpec(shape=[None], dtype="int64")  # src length
        ])
    # Save converted static graph model
    paddle.jit.save(model, args.export_path)


if __name__ == "__main__":
    main()
