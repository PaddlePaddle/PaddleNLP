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
import sys
from pprint import pprint

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer

sys.path.append('./')

from utils import postprocess, get_label_maps, create_dataloader, extract_events

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--task_type", choices=['relation_extraction', 'event_extraction', 'entity_extraction', 'opinion_extraction'], default="entity_extraction", type=str, help="Select the training task type.")
parser.add_argument("--label_maps_path", default="./ner_data/label_maps.json", type=str, help="The file path of the labels dictionary.")
parser.add_argument("--model_path_prefix", type=str, required=True, default='./export/inference', help="The path to model info in static graph.")
parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def reader(inputs):
    for x in inputs:
        yield {"text": x}


class Predictor(object):
    """Predictor"""

    def __init__(self, model_path_prefix, device, task_type, label_maps):
        self.task_type = task_type
        self.label_maps = label_maps

        model_file = model_path_prefix + ".pdmodel"
        params_file = model_path_prefix + ".pdiparams"
        config = paddle.inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]

        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])

    def predict(self, dataloader):
        all_preds = ([], []) if self.task_type in [
            "opinion_extraction", "relation_extraction"
        ] else []
        for batch in dataloader:
            input_ids, attention_masks, offset_mappings, texts = batch
            self.input_handles[0].copy_from_cpu(
                input_ids.numpy().astype('int64'))
            self.input_handles[1].copy_from_cpu(
                attention_masks.numpy().astype('int64'))
            self.predictor.run()
            logits = paddle.to_tensor(self.output_handle.copy_to_cpu())
            batch_outputs = postprocess(logits, offset_mappings, texts,
                                        input_ids.shape[1], self.label_maps,
                                        self.task_type)
            if isinstance(batch_outputs, tuple):
                all_preds[0].extend(batch_outputs[0])  # Entity output
                all_preds[1].extend(batch_outputs[1])  # Relation output
            else:
                all_preds.extend(batch_outputs)
        return all_preds


if __name__ == "__main__":
    label_maps = get_label_maps(args.task_type, args.label_maps_path)

    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-base-zh')

    # Define predictor to do prediction.
    predictor = Predictor(args.model_path_prefix, args.device, args.task_type,
                          label_maps)
    all_preds = predictor.predict(infer_dataloader)
    # Entity extraction sample
    input_list = [
        "金石开：阿森纳的伤病还是对球队有一定的影响，尤其是杜尔的缺席，让老将西尔维斯特必须要出任主力，",
        "所以大多数人都是从巴厘岛南部开始环岛之旅。"
    ]

    # Relation extraction sample
    # input_list = ["歌曲《墨写你的美》是由歌手冷漠演唱的一首歌曲", "常山公主是晋文帝司马昭的女儿，母不详，从小双目失明"]

    # Event extraction sample
    # input_list = ['8月31日，第四届两岸关系天府论坛在四川眉山市举行', '国际金价短期回调 后市银价有望出现较大涨幅']

    # Opinion extraction sample
    # input_list = ["环境也很好，店里的人很热情，所以我办了卡，", "店位于火车站附近，交通便利，酒店服务人员热情"]

    infer_ds = load_dataset(reader, inputs=input_list, lazy=False)

    infer_dataloader = create_dataloader(infer_ds,
                                         tokenizer,
                                         max_seq_len=args.max_seq_len,
                                         batch_size=args.batch_size,
                                         label_maps=label_maps,
                                         mode="infer",
                                         task_type=args.task_type)

    if args.task_type == "entity_extraction":
        for ent, text in zip(all_preds, input_list):
            print("1. Input text: ")
            print(text)
            print("2. Result: ")
            print(ent)
            print("-----------------------------")
    elif args.task_type == "relation_extraction":
        _, rel_preds = all_preds
        for rel, text in zip(rel_preds, input_list):
            print("1. Input text: ")
            print(text)
            print("2. SPO list result: ")
            pprint(rel)
            print("-----------------------------")
    elif args.task_type == "event_extraction":
        for pred_events, text in zip(all_preds, input_list):
            print("1. Input text: ")
            print(text)
            print("2. Result: ")
            event_list = extract_events(pred_events, text)
            pprint(event_list)
            print("-----------------------------")
    elif args.task_type == "opinion_extraction":
        ent_preds, rel_preds = all_preds
        for ent, rel, text in zip(ent_preds, rel_preds, input_list):
            print("1. Input text: ")
            print(text)
            print("2. Aspect and Opinion result: ")
            pprint(ent)
            print("3. ASO result: ")
            pprint(rel)
            print("-----------------------------")
