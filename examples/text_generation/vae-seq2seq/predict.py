#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io

import numpy as np
import paddle

from model import VAESeq2SeqInferModel
from args import parse_args
from data import get_vocab


def infer():
    tar_id2vocab, BOS_ID, EOS_ID = get_vocab(args.dataset, args.batch_size)
    vocab_size = len(tar_id2vocab)

    print(args)
    net = VAESeq2SeqInferModel(args.embed_dim, args.hidden_size,
                               args.latent_size, vocab_size)

    model = paddle.Model(net)
    model.prepare()
    model.load(args.init_from_ckpt)

    infer_output = paddle.ones((args.batch_size, 1), dtype='int64') * BOS_ID

    space_token = ' '
    line_token = '\n'
    with io.open(args.infer_output_file, 'w', encoding='utf-8') as out_file:
        predict_lines = model.predict_batch(infer_output)[0]
        for line in predict_lines:
            end_id = -1
            if EOS_ID in line:
                end_id = np.where(line == EOS_ID)[0][0]
            new_line = [tar_id2vocab[e[0]] for e in line[:end_id]]
            out_file.write(space_token.join(new_line))
            out_file.write(line_token)


if __name__ == '__main__':
    args = parse_args()
    infer()
