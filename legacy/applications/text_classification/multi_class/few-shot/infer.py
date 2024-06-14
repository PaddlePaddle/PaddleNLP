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
import json
import os

import numpy as np
import onnxruntime as ort
import paddle2onnx
import psutil
import six

from paddlenlp.prompt import AutoTemplate, PromptDataCollatorWithPadding
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
parser.add_argument("--model_name", default="ernie-3.0-base-zh", type=str, help="The name of pretrained model.")
parser.add_argument("--data_dir", default=None, type=str, help="The path to the prediction data, including label.txt and data.txt.")
parser.add_argument("--max_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--use_fp16", action='store_true', help="Whether to use fp16 inference, only takes effect when deploying on gpu.")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads", default=psutil.cpu_count(logical=False), type=int, help="num_threads for cpu.")
parser.add_argument("--device", choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--device_id", default=0, help="Select which gpu device to train model.")
args = parser.parse_args()
# yapf: enable


class InferBackend(object):
    def __init__(self, model_path_prefix, device="cpu", device_id=0, use_fp16=False, num_threads=10):

        if not isinstance(device, six.string_types):
            logger.error(
                ">>> [InferBackend] The type of device must be string, but the type you set is: ", type(device)
            )
            exit(0)
        if device not in ["cpu", "gpu"]:
            logger.error(">>> [InferBackend] The device must be cpu or gpu, but your device is set to:", type(device))
            exit(0)

        logger.info(">>> [InferBackend] Creating Engine ...")

        onnx_model = paddle2onnx.command.c_paddle_to_onnx(
            model_file=model_path_prefix + ".pdmodel",
            params_file=model_path_prefix + ".pdiparams",
            opset_version=13,
            enable_onnx_checker=True,
        )
        infer_model_dir = model_path_prefix.rsplit("/", 1)[0]
        float_onnx_file = os.path.join(infer_model_dir, "model.onnx")
        with open(float_onnx_file, "wb", encoding="utf-8") as f:
            f.write(onnx_model)

        if device == "gpu":
            logger.info(">>> [InferBackend] Use GPU to inference ...")
            providers = ["CUDAExecutionProvider"]
            if use_fp16:
                logger.info(">>> [InferBackend] Use FP16 to inference ...")
                import onnx
                from onnxconverter_common import float16

                fp16_model_file = os.path.join(infer_model_dir, "fp16_model.onnx")
                onnx_model = onnx.load_model(float_onnx_file)
                trans_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
                onnx.save_model(trans_model, fp16_model_file)
                onnx_model = fp16_model_file
        else:
            logger.info(">>> [InferBackend] Use CPU to inference ...")
            providers = ["CPUExecutionProvider"]
            if use_fp16:
                logger.warning(
                    ">>> [InferBackend] Ignore use_fp16 as it only " + "takes effect when deploying on gpu..."
                )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        self.predictor = ort.InferenceSession(
            onnx_model, sess_options=sess_options, providers=providers, provider_options=[{"device_id": device_id}]
        )

        if device == "gpu":
            try:
                assert "CUDAExecutionProvider" in self.predictor.get_providers()
            except AssertionError:
                raise AssertionError(
                    "The environment for GPU inference is not set properly. "
                    "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. "
                    "Please run the following commands to reinstall: \n "
                    "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"
                )
        logger.info(">>> [InferBackend] Engine Created ...")

    def infer(self, input_dict: dict):
        result = self.predictor.run(None, input_dict)
        return result


class MultiClassPredictor(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(args.model_name)
        self.template, self.labels, self.input_handles = self.post_init()
        self.collate_fn = PromptDataCollatorWithPadding(
            self.tokenizer, padding=True, return_tensors="np", return_attention_mask=True
        )

        self.inference_backend = InferBackend(
            self.args.model_path_prefix,
            self.args.device,
            self.args.device_id,
            self.args.use_fp16,
            self.args.num_threads,
        )

    def post_init(self):
        export_path = os.path.dirname(self.args.model_path_prefix)
        template_path = os.path.join(export_path, "template_config.json")
        with open(template_path, "r", encoding="utf-8") as fp:
            prompt = json.load(fp)
            template = AutoTemplate.create_from(prompt, self.tokenizer, self.args.max_length, self.model)
        keywords = template.extract_template_keywords(template.prompt)
        inputs = ["input_ids", "token_type_ids", "position_ids", "attention_mask"]
        if "mask" in keywords:
            inputs.append("masked_positions")
        if "soft" in keywords:
            inputs.append("soft_token_ids")
        if "encoder" in keywords:
            inputs.append("encoder_ids")
        verbalizer_path = os.path.join(export_path, "verbalizer_config.json")
        with open(verbalizer_path, "r", encoding="utf-8") as fp:
            label_words = json.load(fp)
            labels = sorted(list(label_words.keys()))

        return template, labels, inputs

    def predict(self, input_data: list):
        encoded_inputs = self.preprocess(input_data)
        infer_result = self.infer_batch(encoded_inputs)
        result = self.postprocess(infer_result)
        self.printer(result, input_data)
        return result

    def _infer(self, input_dict):
        infer_data = self.inference_backend.infer(input_dict)
        return infer_data

    def infer_batch(self, inputs):
        num_sample = len(inputs)
        infer_data = None
        num_infer_data = None
        for index in range(0, num_sample, self.args.batch_size):
            left, right = index, index + self.args.batch_size
            batch_dict = self.collate_fn(inputs[left:right])
            input_dict = {}
            for key in self.input_handles:
                value = batch_dict[key]
                if key == "attention_mask":
                    if value.ndim == 2:
                        value = (1 - value[:, np.newaxis, np.newaxis, :]) * -1e4
                    elif value.ndim != 4:
                        raise ValueError("Expect attention mask with ndim=2 or 4, but get ndim={}".format(value.ndim))
                    value = value.astype("float32")
                else:
                    value = value.astype("int64")
                input_dict[key] = value
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
        text = [{"text_a": x} for x in input_data]
        inputs = [self.template(x) for x in text]
        return inputs

    def postprocess(self, infer_data):
        preds = np.argmax(infer_data[0], axis=-1)
        labels = [self.labels[x] for x in preds]
        return {"label": labels}

    def printer(self, result, input_data):
        label = result["label"]
        for i in range(len(label)):
            logger.info("input data: {}".format(input_data[i]))
            logger.info("labels: {}".format(label[i]))
            logger.info("-----------------------------")


if __name__ == "__main__":
    for arg_name, arg_value in vars(args).items():
        logger.info("{:20}: {}".format(arg_name, arg_value))

    predictor = MultiClassPredictor(args)

    text_dir = os.path.join(args.data_dir, "data.txt")
    with open(text_dir, "r", encoding="utf-8") as f:
        text_list = [x.strip() for x in f.readlines()]

    predictor.predict(text_list)
