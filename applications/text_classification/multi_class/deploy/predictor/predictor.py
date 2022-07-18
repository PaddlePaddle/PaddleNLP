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
import json
import time

import six
import numpy as np
import paddle
import paddle2onnx
import onnxruntime as ort
from sklearn.metrics import f1_score

from paddlenlp.transformers import AutoTokenizer
import paddle.nn.functional as F
from paddlenlp.utils.log import logger


class InferBackend(object):

    def __init__(self,
                 model_path_prefix,
                 device='cpu',
                 device_id=0,
                 use_fp16=False,
                 use_quantize=False,
                 num_threads=10):
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
            if use_quantize:
                logger.info(
                    ">>> [InferBackend] use_quantize only takes effect when deploying on cpu, use_fp16 for acceleration when deploying on gpu ..."
                )
            sess_options = ort.SessionOptions()
            self.predictor = ort.InferenceSession(
                onnx_model,
                sess_options=sess_options,
                providers=['CUDAExecutionProvider'],
                provider_options=[{
                    'device_id': device_id
                }])
            try:
                assert 'CUDAExecutionProvider' in self.predictor.get_providers()
            except AssertionError:
                raise AssertionError(
                    f"The environment for GPU inference is not set properly. "
                    "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. "
                    "Please run the following commands to reinstall: \n "
                    "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"
                )
        else:
            logger.info(">>> [InferBackend] Use CPU to inference ...")
            if use_fp16:
                logger.info(
                    ">>> [InferBackend] use_fp16 only takes effect when deploying on gpu, use_quantize for acceleration when deploying on cpu ..."
                )
            if use_quantize:
                dynamic_quantize_model = os.path.join(infer_model_dir,
                                                      "int8_model.onnx")
                self.dynamic_quantize(float_onnx_file, dynamic_quantize_model)
                onnx_model = dynamic_quantize_model
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = num_threads
            self.predictor = ort.InferenceSession(
                onnx_model,
                sess_options=sess_options,
                providers=['CPUExecutionProvider'])
        input_name1 = self.predictor.get_inputs()[1].name
        input_name2 = self.predictor.get_inputs()[0].name
        self.input_handles = [input_name1, input_name2]

        logger.info(">>> [InferBackend] Engine Created ...")

    def dynamic_quantize(self, input_float_model, dynamic_quantized_model):
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic(input_float_model, dynamic_quantized_model)

    def infer(self, input_dict: dict):
        result = self.predictor.run(None, input_dict)
        return result


class Predictor(object):

    def __init__(self, args, label_list):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                       use_faster=True)
        self.label_list = label_list
        self.batch_size = args.batch_size
        self.max_seq_length = args.max_seq_length

        self.inference_backend = InferBackend(args.model_path_prefix,
                                              args.device, args.device_id,
                                              args.use_fp16, args.use_quantize,
                                              args.num_threads)

    def preprocess(self, input_data: list):

        # tokenizer + pad
        data = self.tokenizer(input_data,
                              max_length=self.max_seq_length,
                              padding=True,
                              truncation=True)

        return {
            "input_ids": np.array(data["input_ids"], dtype="int64"),
            "token_type_ids": np.array(data["token_type_ids"], dtype="int64")
        }

    def postprocess(self, infer_data):

        max_value = np.max(infer_data, axis=1, keepdims=True)
        exp_data = np.exp(infer_data - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        labels = probs.argmax(axis=-1)
        return labels

    def infer(self, data):
        infer_data = self.inference_backend.infer(data)
        logits = np.array(infer_data[0])
        return logits

    def infer_batch(self, preprocess_result):
        sample_num = len(preprocess_result["input_ids"])
        infer_result = None
        for i in range(0, sample_num, self.batch_size):
            batch_size = min(self.batch_size, sample_num - i)
            input_ids = [
                preprocess_result["input_ids"][i + j] for j in range(batch_size)
            ]
            token_type_ids = [
                preprocess_result["token_type_ids"][i + j]
                for j in range(batch_size)
            ]
            preprocess_result_batch = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids
            }
            result = self.infer(preprocess_result_batch)
            if infer_result is None:
                infer_result = result
            else:
                infer_result = np.append(infer_result, result, axis=0)
        return infer_result

    def printer(self, result, input_data):

        for idx, text in enumerate(input_data):
            labels = []
            logger.info("input data: {}".format(text))
            logger.info("labels: {}".format(self.label_list[result[idx]]))
            logger.info('----------------------------')

    def predict(self, input_data: list):
        preprocess_result = self.preprocess(input_data)
        infer_result = self.infer_batch(preprocess_result)
        result = self.postprocess(infer_result)
        self.printer(result, input_data)
        return

    def performance(self, preprocess_result):
        nums = len(preprocess_result["input_ids"])

        start = time.time()
        infer_result = self.infer_batch(preprocess_result)
        total_time = time.time() - start
        logger.info("sample nums: %s, time: %.2f, latency: %.2f ms" %
                    (nums, total_time, 1000 * total_time / nums))
        return

    def evaluate(self, preprocess_result, labels):

        infer_result = self.infer_batch(preprocess_result)
        max_value = np.max(infer_result, axis=1, keepdims=True)
        exp_data = np.exp(infer_result - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        preds = probs.argmax(axis=-1)
        acc = np.mean(np.equal(preds, labels))
        logger.info("acc: %.2f" % (acc * 100))
        return

    def get_text_and_label(self, ds):
        """
        Return text and label list
        """
        all_texts = []
        all_labels = []
        for ii in range(len(ds)):
            all_texts.append(ds[ii]['text_a'])
            all_labels.append(ds[ii]['label'])
        return all_texts, np.array(all_labels, dtype='int64')
