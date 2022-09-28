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
import tempfile

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddlenlp.transformers import GPTChineseTokenizer, GPTTokenizer

MODEL_CLASSES = {
    "gpt-cn": (GPTChineseTokenizer, ),
    "gpt": (GPTTokenizer, ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True, help="The path prefix of inference model to be used.")
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str,
                        help="Path to tokenizer or shortcut name selected in the list: "
                             + ", ".join(sum([
                                list(classes[-1].pretrained_init_configuration.keys())
                                for classes in MODEL_CLASSES.values()], [])), )
    # yapf: enable

    args = parser.parse_args()
    return args


class Predictor(object):

    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        nranks, rank = fleet.worker_num(), fleet.worker_index()
        trainer_endpoints = fleet.worker_endpoints()
        current_endpoint = trainer_endpoints[rank]

        model_path = os.path.join(args.model_path, 'rank_' + str(rank))
        if not os.path.isdir(model_path):
            config = paddle.inference.Config(args.model_path + '.pdmodel',
                                             args.model_path + '.pdiparams')
        else:
            model_file = None
            param_file = None
            for f in os.listdir(model_path):
                if '.pdmodel' in f:
                    assert model_file is None
                    model_file = os.path.join(model_path, f)
                if '.pdiparams' in f:
                    assert param_file is None
                    param_file = os.path.join(model_path, f)
            config = paddle.inference.Config(model_file, param_file)

        config.enable_use_gpu(100, device_id)
        config.switch_use_feed_fetch_ops(False)

        dist_config = config.dist_config()
        dist_config.enable_dist_model(True)
        dist_config.set_ranks(nranks, rank)
        dist_config.set_endpoints(trainer_endpoints, current_endpoint)

        # Difficult to use, needs to be simplified...
        ring_id_to_ranks = ','.join(['0'] + [str(i) for i in range(nranks)])
        rank_to_ring_ids = ''
        for i in range(nranks):
            rank_to_ring_ids += '{},0\n'.format(i)
        comm_config_str = '[ring_id -> ranks]\n' + ring_id_to_ranks + '\n[rank -> ring_ids]\n' + rank_to_ring_ids

        # Use temp file so that each rank will not have RW conflicts
        with tempfile.NamedTemporaryFile('w') as f:
            f.write(comm_config_str)
            f.seek(0)  # Move to beginning
            dist_config.set_comm_init_config(f.name)

            config.set_dist_config(dist_config)
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

    def predict(self, dataset, batch_size=1):
        outputs = []
        for data in dataset:
            output = self.predict_batch(data)
            outputs.append(output)
        return outputs


def main():
    paddle.enable_static()
    args = parse_args()

    fleet.init(is_collective=True)
    predictor = Predictor.create_predictor(args)
    args.model_type = args.model_type.lower()
    tokenizer_class, = MODEL_CLASSES[args.model_type]

    if args.tokenizer_name_or_path is None:
        if args.model_type == "gpt":
            tokenizer = tokenizer_class.from_pretrained('gpt2-en')
        elif args.model_type == "gpt-cn":
            tokenizer = tokenizer_class.from_pretrained('gpt-cpm-large-cn')
    else:
        tokenizer = tokenizer_class.from_pretrained(
            os.path.dirname(args.tokenizer_name_or_path))

    if args.model_type == "gpt":
        text = [
            "Question: Who is the CEO of Apple? Answer:",
            "Question: Who is the CEO of Facebook? Answer:",
            "Question: How tall is the highest peak in the world? Answer:",
            "Question: Who is the president of the united states? Answer:",
            "Question: Where is the capital of France? Answer:",
            "Question: What is the largest animal in the ocean? Answer:",
            "Question: Who is the chancellor of Germany? Answer:",
        ]
    elif args.model_type == "gpt-cn":
        text = [
            "问题：苹果的CEO是谁? 答案：",
            "问题：中国的首都是哪里？答案：",
            "问题：世界上最高的山峰是? 答案：",
        ]
    inputs = tokenizer(text,
                       padding=True,
                       return_attention_mask=True,
                       return_position_ids=True)
    ids = np.array(inputs["input_ids"]).reshape(len(text), -1).astype('int64')
    attention_mask = np.array(inputs["attention_mask"]).reshape(
        len(text), -1).astype('float32')
    position_ids = np.array(inputs["position_ids"]).reshape(len(text),
                                                            -1).astype('int64')

    dataset = [[ids, attention_mask, position_ids]]

    outs = predictor.predict(dataset)
    for out in outs:
        for i in range(out[0].shape[0]):
            out_ids = [int(x) for x in out[0][i]]
            ret_str = tokenizer.convert_ids_to_string(out_ids)
            ret_str = text[i] + ret_str
            print(ret_str)


if __name__ == "__main__":
    main()
