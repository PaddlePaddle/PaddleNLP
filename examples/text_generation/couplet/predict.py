# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import io
from args import parse_args

import numpy as np
import paddle
from paddlenlp.data import Vocab

from data import create_infer_loader
from model import Seq2SeqAttnInferModel


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def do_predict(args):
    device = paddle.set_device(args.device)

    test_loader, vocab = create_infer_loader(args.batch_size)
    vocab_size = len(vocab)
    pad_id = vocab[vocab.eos_token]
    bos_id = vocab[vocab.bos_token]
    eos_id = vocab[vocab.eos_token]
    trg_idx2word = vocab.idx_to_token

    model = paddle.Model(
        Seq2SeqAttnInferModel(vocab_size,
                              args.hidden_size,
                              args.hidden_size,
                              args.num_layers,
                              bos_id=bos_id,
                              eos_id=eos_id,
                              beam_size=args.beam_size,
                              max_out_len=256))

    model.prepare()

    # Load the trained model
    assert args.init_from_ckpt, (
        "Please set reload_model to load the infer model.")
    model.load(args.init_from_ckpt)

    # TODO(guosheng): use model.predict when support variant length
    with io.open(args.infer_output_file, 'w', encoding='utf-8') as f:
        for data in test_loader():
            inputs = data[:2]
            finished_seq = model.predict_batch(inputs=list(inputs))[0]
            finished_seq = finished_seq[:, :, np.newaxis] if len(
                finished_seq.shape) == 2 else finished_seq
            finished_seq = np.transpose(finished_seq, [0, 2, 1])
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    id_list = post_process_seq(beam, bos_id, eos_id)
                    word_list = [trg_idx2word[id] for id in id_list]
                    sequence = "\x02".join(word_list) + "\n"
                    f.write(sequence)
                    break


if __name__ == "__main__":
    args = parse_args()
    do_predict(args)
