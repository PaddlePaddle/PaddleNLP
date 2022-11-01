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

import os
import six
import psutil
import argparse

import numpy as np

from paddlenlp.utils.log import logger
from paddlenlp.prompt import AutoTemplate, Verbalizer, InputExample
from paddlenlp.transformers import AutoTokenizer
import paddle2onnx
import onnxruntime as ort

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
parser.add_argument("--model_name_or_path", default="ernie-3.0-base-zh", type=str, help="The directory or name of model.")
parser.add_argument("--data_dir", default=None, type=str, help="The path to the prediction data, including label.txt and data.txt.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--use_fp16", action='store_true', help="Whether to use fp16 inference, only takes effect when deploying on gpu.")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads", default=psutil.cpu_count(logical=False), type=int, help="num_threads for cpu.")
parser.add_argument("--device", choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--device_id", default=0, help="Select which gpu device to train model.")
args = parser.parse_args()
# yapf: enable


class InferBackend(object):

    def __init__(self,
                 model_path_prefix,
                 device="cpu",
                 device_id=0,
                 use_fp16=False,
                 num_threads=10):

        if not isinstance(device, six.string_types):
            logger.error(
                ">>> [InferBackend] The type of device must be string, but the type you set is: ",
                type(device))
            exit(0)
        if device not in ['cpu', 'gpu']:
            logger.error(
                ">>> [InferBackend] The device must be cpu or gpu, but your device is set to:",
                type(device))
            exit(0)

        logger.info(">>> [InferBackend] Creating Engine ...")

        onnx_model = paddle2onnx.command.c_paddle_to_onnx(
            model_file=model_path_prefix + ".pdmodel",
            params_file=model_path_prefix + ".pdiparams",
            opset_version=13,
            enable_onnx_checker=True)
        infer_model_dir = model_path_prefix.rsplit("/", 1)[0]
        float_onnx_file = os.path.join(infer_model_dir, "model.onnx")
        with open(float_onnx_file, "wb") as f:
            f.write(onnx_model)

        if device == "gpu":
            logger.info(">>> [InferBackend] Use GPU to inference ...")
            providers = ['CUDAExecutionProvider']
            if use_fp16:
                logger.info(">>> [InferBackend] Use FP16 to inference ...")
                from onnxconverter_common import float16
                import onnx
                fp16_model_file = os.path.join(infer_model_dir,
                                               "fp16_model.onnx")
                onnx_model = onnx.load_model(float_onnx_file)
                trans_model = float16.convert_float_to_float16(
                    onnx_model, keep_io_types=True)
                onnx.save_model(trans_model, fp16_model_file)
                onnx_model = fp16_model_file
        else:
            logger.info(">>> [InferBackend] Use CPU to inference ...")
            providers = ['CPUExecutionProvider']
            if use_fp16:
                logger.warning(
                    ">>> [InferBackend] Ignore use_fp16 as it only " +
                    "takes effect when deploying on gpu...")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        self.predictor = ort.InferenceSession(onnx_model,
                                              sess_options=sess_options,
                                              providers=providers,
                                              provider_options=[{
                                                  'device_id':
                                                  device_id
                                              }])
        self.input_handles = [
            self.predictor.get_inputs()[0].name,
            self.predictor.get_inputs()[1].name,
            self.predictor.get_inputs()[2].name
        ]

        if device == "gpu":
            try:
                assert 'CUDAExecutionProvider' in self.predictor.get_providers()
            except AssertionError:
                raise AssertionError(
                    f"The environment for GPU inference is not set properly. "
                    "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. "
                    "Please run the following commands to reinstall: \n "
                    "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"
                )
        logger.info(">>> [InferBackend] Engine Created ...")

    def infer(self, input_dict: dict):
        input_dict = {
            k: v
            for k, v in input_dict.items() if k in self.input_handles
        }
        result = self.predictor.run(None, input_dict)
        return result


class MultiLabelPredictor(object):

    def __init__(self, args, label_list):
        self._label_list = label_list
        self._tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self._max_seq_length = args.max_seq_length
        self._batch_size = args.batch_size
        self.inference_backend = InferBackend(args.model_path_prefix,
                                              args.device, args.device_id,
                                              args.use_fp16, args.num_threads)
        self._template = AutoTemplate.load_from(
            os.path.dirname(args.model_path_prefix), self._tokenizer,
            args.max_seq_length)

    def predict(self, input_data: list):
        encoded_inputs = self.preprocess(input_data)
        infer_result = self.infer_batch(encoded_inputs)
        result = self.postprocess(infer_result)
        self.printer(result, input_data)
        return result

    def _infer(self, input_dict):
        infer_data = self.inference_backend.infer(input_dict)
        return infer_data

    def infer_batch(self, encoded_inputs):
        num_sample = len(encoded_inputs["input_ids"])
        infer_data = None
        num_infer_data = None
        for idx in range(0, num_sample, self._batch_size):
            l, r = idx, idx + self._batch_size
            keys = encoded_inputs.keys()
            input_dict = {k: encoded_inputs[k][l:r] for k in keys}
            results = self._infer(input_dict)
            if infer_data is None:
                infer_data = [[x] for x in results]
                num_infer_data = len(results)
            else:
                for i in range(num_infer_data):
                    infer_data[i].append(results[i])
        for i in range(num_infer_data):
            infer_data[i] = np.concatenate(infer_data[i], axis=0)
        return infer_data

    def preprocess(self, input_data: list):
        text = [InputExample(text_a=x) for x in input_data]
        inputs = [self._template.wrap_one_example(x) for x in text]
        inputs = {
            "input_ids":
            np.array([x["input_ids"] for x in inputs], dtype="int64"),
            "mask_ids":
            np.array([x["mask_ids"] for x in inputs], dtype="int64"),
            "soft_token_ids":
            np.array([x["soft_token_ids"] for x in inputs], dtype="int64")
        }
        return inputs

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def postprocess(self, infer_data):
        threshold = 0.5
        probs = self.sigmoid(infer_data[0])
        label_ids = np.argwhere(probs > threshold)
        labels = [[] for _ in range(probs.shape[0])]
        for idx, label_id in label_ids:
            labels[idx].append(self._label_list[label_id])
        return {"label": labels}

    def printer(self, result, input_data):
        label = result["label"]
        for i in range(len(label)):
            logger.info("input data: {}".format(input_data[i]))
            logger.info("labels: {}".format(", ".join(label[i])))
            logger.info("-----------------------------")


if __name__ == "__main__":
    for arg_name, arg_value in vars(args).items():
        logger.info("{:20}: {}".format(arg_name, arg_value))

    export_path = os.path.dirname(args.model_path_prefix)
    labels, _ = Verbalizer.load_from(export_path)

    text_dir = os.path.join(args.data_dir, "data.txt")
    with open(text_dir, "r", encoding="utf-8") as f:
        text_list = [x.strip() for x in f.readlines()]

    predictor = MultiLabelPredictor(args, labels)
    predictor.predict(text_list)
