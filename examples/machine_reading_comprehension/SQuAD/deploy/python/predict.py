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
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from functools import partial

import paddle
from paddle import inference
from paddle.io import DataLoader
from datasets import load_dataset
from paddlenlp.data import Pad, Stack, Dict
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction

from args import parse_args
from run_squad import MODEL_CLASSES, prepare_validation_features


class Predictor(object):

    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(args.model_name_or_path + ".pdmodel",
                                         args.model_name_or_path + ".pdiparams")
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

    def predict(self, dataset, raw_dataset, collate_fn, args, do_eval=True):
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)
        data_loader = paddle.io.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           collate_fn=collate_fn,
                                           num_workers=0,
                                           return_list=True)
        outputs = []
        all_start_logits = []
        all_end_logits = []
        for data in data_loader:
            output = self.predict_batch(data)
            outputs.append(output)
            if do_eval:
                all_start_logits.extend(list(output[0]))
                all_end_logits.extend(list(output[1]))
        if do_eval:
            all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
                raw_dataset, data_loader.dataset,
                (all_start_logits, all_end_logits),
                args.version_2_with_negative, args.n_best_size,
                args.max_answer_length, args.null_score_diff_threshold)
            squad_evaluate(examples=[raw_data for raw_data in raw_dataset],
                           preds=all_predictions,
                           na_probs=scores_diff_json)
        return outputs


def main():
    args = parse_args()

    predictor = Predictor.create_predictor(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        os.path.dirname(args.model_name_or_path))

    if args.version_2_with_negative:
        raw_dataset = load_dataset('squad_v2', split='validation')
    else:
        raw_dataset = load_dataset('squad', split='validation')
    column_names = raw_dataset.column_names
    dataset = raw_dataset.map(partial(prepare_validation_features,
                                      tokenizer=tokenizer,
                                      args=args),
                              batched=True,
                              remove_columns=column_names,
                              num_proc=4)

    batchify_fn = lambda samples, fn=Dict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        }): fn(samples)
    predictor = Predictor.create_predictor(args)
    predictor.predict(dataset, raw_dataset, args=args, collate_fn=batchify_fn)


if __name__ == "__main__":
    main()
