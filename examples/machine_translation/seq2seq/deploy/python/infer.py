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
import io
import os
import sys

sys.path.append("../../")

from functools import partial
import numpy as np

import paddle
from paddle import inference
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import BLEU

from args import parse_args
from data import create_infer_loader
from predict import post_process_seq
from paddlenlp.data import Vocab


class Predictor(object):

    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(args.export_path + ".pdmodel",
                                         args.export_path + ".pdiparams")
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        elif args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handles = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]
        return cls(predictor, input_handles, output_handles)

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field.numpy(
            ) if isinstance(input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def predict(self, dataloader, infer_output_file, trg_idx2word, bos_id,
                eos_id):
        cand_list = []
        with io.open(infer_output_file, 'w', encoding='utf-8') as f:
            for data in dataloader():
                finished_seq = self.predict_batch(data)[0]
                finished_seq = finished_seq[:, :, np.newaxis] if len(
                    finished_seq.shape) == 2 else finished_seq
                finished_seq = np.transpose(finished_seq, [0, 2, 1])
                for ins in finished_seq:
                    for beam_idx, beam in enumerate(ins):
                        id_list = post_process_seq(beam, bos_id, eos_id)
                        word_list = [trg_idx2word[id] for id in id_list]
                        sequence = " ".join(word_list) + "\n"
                        f.write(sequence)
                        cand_list.append(word_list)
                        break

        test_ds = load_dataset('iwslt15', splits='test')
        bleu = BLEU()
        for i, data in enumerate(test_ds):
            ref = data['vi'].split()
            bleu.add_inst(cand_list[i], [ref])
        print("BLEU score is %s." % bleu.score())


def main():
    args = parse_args()

    predictor = Predictor.create_predictor(args)
    test_loader, src_vocab_size, tgt_vocab_size, bos_id, eos_id = create_infer_loader(
        args)
    tgt_vocab = Vocab.load_vocabulary(**test_loader.dataset.vocab_info['vi'])
    trg_idx2word = tgt_vocab.idx_to_token

    predictor.predict(test_loader, args.infer_output_file, trg_idx2word, bos_id,
                      eos_id)


if __name__ == "__main__":
    main()
