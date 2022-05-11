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

from pathlib import Path
import six
import argparse
import os
import time
import sys
from functools import partial
import distutils.util
import numpy as np

import paddle
from paddle.metric import Accuracy
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import AutoTokenizer


class InferBackend(object):
    def __init__(self,
                 model_path,
                 batch_size=32,
                 device='cpu',
                 use_fp16=False,
                 enable_quantize=False,
                 set_dynamic_shape=False,
                 num_threads=10):
        file_name = model_path.split('/')[-1]
        model_dir = model_path[:-1 * len(file_name)]
        int8_model = self.paddle_quantize_model(
            model_dir, file_name + ".pdmodel", file_name + ".pdiparams")
        print(">>> [InferBackend] creat engine ...")
        if device == 'gpu' and int8_model or use_fp16:
            from paddle import inference
            import paddle
            config = paddle.inference.Config(model_path + ".pdmodel",
                                             model_path + ".pdiparams")
            config.enable_use_gpu(100, 0)
            paddle.set_device("gpu")
            if int8_model and use_fp16:
                print(
                    ">>> [InferBackend] load a paddle quantize model, use_fp16 has been closed..."
                )
                use_fp16 = False

            if use_fp16:
                assert device == 'gpu', "When use_fp16, please set device to gpu and install requirement_gpu.txt."
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Half,
                    max_batch_size=batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            else:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Int8,
                    max_batch_size=batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            shape_file = "shape_info.txt"
            if set_dynamic_shape:
                config.collect_shape_range_info(shape_file)
            else:
                config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)
            config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
            self.predictor = paddle.inference.create_predictor(config)
            self.input_handles = [
                self.predictor.get_input_handle(name)
                for name in self.predictor.get_input_names()
            ]
            self.output_handles = [
                self.predictor.get_output_handle(name)
                for name in self.predictor.get_output_names()
            ]
        else:
            import paddle2onnx
            import onnxruntime as ort
            import copy
            import os
            float_onnx_file = "model.onnx"
            paddle2onnx.py_program2onnx(
                model_dir=model_dir,
                model_filename=file_name + ".pdmodel",
                params_filename=file_name + ".pdiparams",
                save_file=float_onnx_file,
                opset_version=13,
                enable_onnx_checker=True)
            dynamic_quantize_onnx_file = copy.copy(float_onnx_file)
            providers = ['CUDAExecutionProvider']
            if enable_quantize:
                dynamic_quantize_onnx_file = "dynamic_quantize_model.onnx"
                self.dynamic_quantize(float_onnx_file,
                                      dynamic_quantize_onnx_file)
                providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.optimized_model_filepath = "./optimize_model.onnx"
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
            self.predictor = ort.InferenceSession(
                dynamic_quantize_onnx_file,
                sess_options=sess_options,
                providers=providers)
            input_name1 = self.predictor.get_inputs()[0].name
            input_name2 = self.predictor.get_inputs()[1].name
            self.input_handles = [input_name1, input_name2]
            self.output_handles = []
        print(">>> [InferBackend] engine created ...")

    def dynamic_quantize(self, input_float_model, dynamic_quantized_model):
        from onnxruntime.quantization import QuantizationMode, quantize_dynamic
        quantize_dynamic(input_float_model, dynamic_quantized_model)

    def paddle_quantize_model(self, model_dir, model_file, params_file):
        import paddle.fluid as fluid
        paddle.enable_static()
        exe = fluid.Executor(fluid.CPUPlace())
        if model_file is None and params_file is None:
            [program, feed_var_names,
             fetch_vars] = fluid.io.load_inference_model(model_dir, exe)
        else:
            [program, feed_var_names,
             fetch_vars] = fluid.io.load_inference_model(
                 model_dir,
                 exe,
                 model_filename=model_file,
                 params_filename=params_file)
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type.count("quantize"):
                    return True
        return False

    def infer(self, data):
        if isinstance(self.predictor,
                      paddle.fluid.core_avx.PaddleInferPredictor):
            for input_field, input_handle in zip(data, self.input_handles):
                input_handle.copy_from_cpu(input_field)
            self.predictor.run()
            output = [
                output_handle.copy_to_cpu()
                for output_handle in self.output_handles
            ]
            return output
        input_dict = {}
        for input_field, input_handle in zip(data, self.input_handles):
            input_dict[input_handle] = input_field
        result = self.predictor.run(None, input_dict)
        return result


METRIC_CLASSES = {
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}


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


class ErniePredictor(object):
    def __init__(self, args):
        if not isinstance(args.device, six.string_types):
            print(
                ">>> [InferBackend] The type of device must be string, but the type you set is: ",
                type(device))
            exit(0)
        args.device = args.device.lower()
        if args.device not in ['cpu', 'gpu']:
            print(
                ">>> [InferBackend] The device must be cpu or gpu, but your device is set to:",
                type(args.device))
            exit(0)
        if args.device == 'cpu':
            args.use_fp16 = False
            args.set_dynamic_shape = False
        if args.device == 'gpu':
            args.num_threads = 10
            args.enable_quantize = False
        self.inference_backend = InferBackend(
            args.model_path,
            batch_size=args.batch_size,
            device=args.device,
            use_fp16=args.use_fp16,
            enable_quantize=args.enable_quantize,
            set_dynamic_shape=args.set_dynamic_shape,
            num_threads=args.num_threads)
        if args.set_dynamic_shape:
            # If set_dynamic_shape is turned on, all required dynamic shapes will be automatically set according to the batch_size and max_seq_length.
            self.set_dynamic_shape(args.max_seq_length, args.max_seq_length)
            exit(0)

    def set_dynamic_shape(self, max_seq_length, batch_size):
        min_batch_size, max_batch_size, opt_batch_size = 1, batch_size, batch_size
        min_seq_len, max_seq_len, opt_seq_len = 2, args.max_seq_length, 32
        batches = [
            [
                np.zeros(
                    [min_batch_size, min_seq_len], dtype="int64"), np.zeros(
                        [min_batch_size, min_seq_len], dtype="int64")
            ],
            [
                np.zeros(
                    [max_batch_size, max_seq_len], dtype="int64"), np.zeros(
                        [max_batch_size, max_seq_len], dtype="int64")
            ],
            [
                np.zeros(
                    [opt_batch_size, opt_seq_len], dtype="int64"), np.zeros(
                        [opt_batch_size, opt_seq_len], dtype="int64")
            ],
        ]
        for batch in batches:
            self.inference_backend.infer(batch)
        print(
            "[InferBackend] Set dynamic shape finished, please close set_dynamic_shape and restart."
        )

    def predict_batch(self, data):
        return self.inference_backend.infer(data)

    def predict(self, dataset, tokenizer, batchify_fn, args):
        paddle.disable_static()
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
            times = []
            for batch in batches:
                input_ids, segment_ids, _ = batchify_fn(batch)
                time1 = time.time()
                output = self.predict_batch([input_ids, segment_ids])
                times.append(time.time() - time1)

            print("task name: %s, mean time: %s, std time: %s" %
                  (args.task_name, np.mean(times) * 1000, np.std(times) * 1000))

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
