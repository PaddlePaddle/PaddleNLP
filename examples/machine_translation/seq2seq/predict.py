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

import io
import numpy as np

import paddle
from args import parse_args

from seq2seq_attn import Seq2SeqAttnInferModel
from data import create_infer_loader
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import BLEU
from paddlenlp.data import Vocab


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

    test_loader, src_vocab_size, tgt_vocab_size, bos_id, eos_id = create_infer_loader(
        args)
    tgt_vocab = Vocab.load_vocabulary(**test_loader.dataset.vocab_info['vi'])

    model = paddle.Model(
        Seq2SeqAttnInferModel(src_vocab_size,
                              tgt_vocab_size,
                              args.hidden_size,
                              args.hidden_size,
                              args.num_layers,
                              args.dropout,
                              bos_id=bos_id,
                              eos_id=eos_id,
                              beam_size=args.beam_size,
                              max_out_len=256))

    model.prepare()

    # Load the trained model
    assert args.init_from_ckpt, (
        "Please set reload_model to load the infer model.")
    model.load(args.init_from_ckpt)

    cand_list = []
    with io.open(args.infer_output_file, 'w', encoding='utf-8') as f:
        for data in test_loader():
            with paddle.no_grad():
                finished_seq = model.predict_batch(inputs=data)[0]
            finished_seq = finished_seq[:, :, np.newaxis] if len(
                finished_seq.shape) == 2 else finished_seq
            finished_seq = np.transpose(finished_seq, [0, 2, 1])
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    id_list = post_process_seq(beam, bos_id, eos_id)
                    word_list = [tgt_vocab.to_tokens(id) for id in id_list]
                    sequence = " ".join(word_list) + "\n"
                    f.write(sequence)
                    cand_list.append(word_list)
                    break

    bleu = BLEU()
    for i, data in enumerate(test_loader.dataset.data):
        ref = data['vi'].split()
        bleu.add_inst(cand_list[i], [ref])
    print("BLEU score is %s." % bleu.score())


if __name__ == "__main__":
    args = parse_args()
    do_predict(args)
