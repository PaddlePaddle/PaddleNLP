# C#opyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import time
import sys
from functools import partial
import distutils.util
import numpy as np
import onnxruntime as ort

import paddle
from paddle import inference
from paddle.metric import Accuracy
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from paddlenlp.transformers import AutoTokenizer

METRIC_CLASSES = {
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='tnews',
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-3.0-medium-zh",
        type=str,
        help="The directory or name of model.", )
    parser.add_argument(
        "--model_path",
        default='tnews_quant_models/mse4/int8',
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu", "xpu"],
        help="Device selected for inference.", )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.", )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--perf_warmup_steps",
        default=20,
        type=int,
        help="Warmup steps for performance test.", )
    parser.add_argument(
        "--use_trt",
        action='store_true',
        help="Whether to use inference engin TensorRT.", )
    parser.add_argument(
        "--perf",
        action='store_true',
        help="Whether to test performance.", )
    parser.add_argument(
        "--collect_shape",
        action='store_true',
        help="Whether to collect shape info.", )
    parser.add_argument(
        "--int8",
        action='store_true',
        help="Whether to use int8 inference.", )
    parser.add_argument(
        "--use_onnxruntime",
        type=distutils.util.strtobool,
        default=False,
        help="Use onnxruntime to infer or not.")
    args = parser.parse_args()
    return args


def convert_example(example,
                    tokenizer,
                    label_list,
                    is_test=False,
                    max_seq_length=512):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = np.array(example["label"], dtype="int64")
    # Convert raw text to feature
    if 'keyword' in example:  # CSL
        sentence1 = " ".join(example['keyword'])
        example = {
            'sentence1': sentence1,
            'sentence2': example['abst'],
            'label': example['label']
        }
    elif 'target' in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
            'target']['span1_text'], example['target']['span2_text'], example[
                'target']['span1_index'], example['target']['span2_index']
        text_list = list(text)
        assert text[pronoun_idx:(pronoun_idx + len(pronoun)
                                 )] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx + len(query)
                               )] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example['sentence'] = text
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length)
    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


@paddle.no_grad()
def evaluate(outputs, metric, data_loader):
    metric.reset()
    for i, batch in enumerate(data_loader):
        input_ids, segment_ids, labels = batch
        logits = paddle.to_tensor(outputs[i][0])
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    print("acc: %s, " % res, end='')


class Predictor(object):
    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        if args.use_onnxruntime:
            sess_options = ort.SessionOptions()
            sess_options.optimized_model_filepath = "./optimize_model.onnx"
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            predictor = ort.InferenceSession(
                args.model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider'])
            input_name1 = predictor.get_inputs()[0].name
            input_name2 = predictor.get_inputs()[1].name
            input_handles = [input_name1, input_name2]
            return cls(predictor, input_handles, [])
        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            cls.device = paddle.set_device("gpu")
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            cls.device = paddle.set_device("cpu")
        elif args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        if args.use_trt:
            if args.int8:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Int8,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            else:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Float32,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            print("Enable TensorRT is: {}".format(
                config.tensorrt_engine_enabled()))
            # Set min/max/opt tensor shape of each trt subgraph input according
            # to dataset.
            # For example, the config of TNEWS data should be 16, 32, 32, 31, 128, 32.
            min_batch_size, max_batch_size, opt_batch_size = 1, 32, 32
            #min_seq_len, max_seq_len, opt_seq_len = 1, 128, 32
            min_seq_len, max_seq_len, opt_seq_len = 1, 512, 32
            min_input_shape = {
                "input_ids": [min_batch_size, min_seq_len],
                "token_type_ids": [min_batch_size, min_seq_len],
                #"full_like_0.tmp_0": [min_batch_size, min_seq_len],
                "full_like_1.tmp_0": [min_batch_size, min_seq_len],
                "tmp_4": [min_batch_size, min_seq_len],
                "unsqueeze2_0.tmp_0": [min_batch_size, 1, 1, min_seq_len]
                #"cast_0.tmp_0": [min_batch_size, 1, 1, min_seq_len]
            }
            max_input_shape = {
                "input_ids": [max_batch_size, max_seq_len],
                "token_type_ids": [max_batch_size, max_seq_len],
                "full_like_1.tmp_0": [max_batch_size, max_seq_len],
                "tmp_4": [max_batch_size, max_seq_len],
                #"cast_0.tmp_0": [max_batch_size, 1, 1, max_seq_len],
                "unsqueeze2_0.tmp_0": [max_batch_size, 1, 1, max_seq_len]
            }
            opt_input_shape = {
                "input_ids": [opt_batch_size, opt_seq_len],
                "token_type_ids": [opt_batch_size, opt_seq_len],
                "full_like_1.tmp_0": [opt_batch_size, opt_seq_len],
                "tmp_4": [opt_batch_size, opt_seq_len],
                #"tmp_0": [opt_batch_size, 1, 1, opt_seq_len]
                #"cast_0.tmp_0": [opt_batch_size, 1, 1, opt_seq_len],
                "unsqueeze2_0.tmp_0": [opt_batch_size, 1, 1, opt_seq_len]
            }

            shape_file = "shape_info.txt"
            if args.collect_shape:
                config.collect_shape_range_info(shape_file)
            else:
                config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)
            #config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
            #                                  opt_input_shape)
        config.delete_pass("embedding_eltwise_layernorm_fuse_pass")

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
        if len(self.output_handles) == 0:
            input_dict = {}
            for input_field, input_handle in zip(data, self.input_handles):
                input_dict[input_handle] = input_field
            result = self.predictor.run(None, input_dict)
            return result

        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def predict(self, dataset, tokenizer, batchify_fn, args):
        batches = [
            dataset[idx:idx + args.batch_size]
            for idx in range(0, len(dataset), args.batch_size)
        ]
        if args.perf:
            for i, batch in enumerate(batches):
                input_ids, segment_ids, label = batchify_fn(batch)
                output = self.predict_batch([input_ids, segment_ids])
                if i > args.perf_warmup_steps:
                    break
            time1 = time.time()
            for batch in batches:
                input_ids, segment_ids, _ = batchify_fn(batch)
                output = self.predict_batch([input_ids, segment_ids])

            print("task name: %s, time: %s, " %
                  (args.task_name, time.time() - time1))

        else:
            metric = METRIC_CLASSES[args.task_name]()
            metric.reset()
            for i, batch in enumerate(batches):
                input_ids, segment_ids, label = batchify_fn(batch)
                output = self.predict_batch([input_ids, segment_ids])
                correct = metric.compute(
                    paddle.to_tensor(output), paddle.to_tensor(label))
                metric.update(correct)

            res = metric.accumulate()
            print("task name: %s, acc: %s, " % (args.task_name, res), end='')


def main():
    paddle.seed(42)
    args = parse_args()

    args.task_name = args.task_name.lower()

    predictor = Predictor.create_predictor(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dev_ds = load_dataset('clue', args.task_name, splits='dev')

    trans_func = partial(
        convert_example,
        label_list=dev_ds.label_list,
        tokenizer=tokenizer,
        is_test=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32")  # label
    ): fn(samples)
    outputs = predictor.predict(dev_ds, tokenizer, batchify_fn, args)


if __name__ == "__main__":
    main()
