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
import time
import numpy as np
from sklearn.metrics import f1_score

import paddle
import paddle2onnx
import onnxruntime as ort
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers import normalize_chars, tokenize_special_chars


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


class EHealthPredictor(object):

    def __init__(self, args, label_list):
        self.label_list = label_list
        self._tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                        use_faster=True)
        self._max_seq_length = args.max_seq_length
        self._batch_size = args.batch_size
        self.inference_backend = InferBackend(args.model_path_prefix,
                                              args.device, args.device_id,
                                              args.use_fp16, args.num_threads)

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

    def performance(self, encoded_inputs):
        nums = len(encoded_inputs["input_ids"])
        start_time = time.time()
        infer_result = self.infer_batch(preprocess_result)
        total_time = time.time() - start_time
        logger.info("sample nums: %d, time: %.2f, latency: %.2f ms" %
                    (nums, total_time, 1000 * total_time / nums))

    def get_text_and_label(self, dataset):
        raise NotImplementedError

    def preprocess(self, input_data: list):
        raise NotImplementedError

    def postprocess(self, infer_data):
        raise NotImplementedError

    def printer(self, result, input_data):
        raise NotImplementedError


class CLSPredictor(EHealthPredictor):

    def preprocess(self, input_data: list):
        norm_text = lambda x: tokenize_special_chars(normalize_chars(x))
        # To deal with a pair of input text.
        if isinstance(input_data[0], list):
            text = [norm_text(sample[0]) for sample in input_data]
            text_pair = [norm_text(sample[1]) for sample in input_data]
        else:
            text = [norm_text(x) for x in input_data]
            text_pair = None

        data = self._tokenizer(text=text,
                               text_pair=text_pair,
                               max_length=self._max_seq_length,
                               padding=True,
                               truncation=True,
                               return_position_ids=True)

        encoded_inputs = {
            "input_ids": np.array(data["input_ids"], dtype="int64"),
            "token_type_ids": np.array(data["token_type_ids"], dtype="int64"),
            "position_ids": np.array(data['position_ids'], dtype="int64")
        }
        return encoded_inputs

    def postprocess(self, infer_data):
        infer_data = infer_data[0]
        max_value = np.max(infer_data, axis=1, keepdims=True)
        exp_data = np.exp(infer_data - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        label = probs.argmax(axis=-1)
        confidence = probs.max(axis=-1)
        return {"label": label, "confidence": confidence}

    def printer(self, result, input_data):
        label, confidence = result["label"], result["confidence"]
        for i in range(len(label)):
            logger.info("input data: {}".format(input_data[i]))
            logger.info("labels: {}, confidence: {}".format(
                self.label_list[label[i]], confidence[i]))
            logger.info("-----------------------------")


class NERPredictor(EHealthPredictor):
    """ The predictor for CMeEE dataset. """
    en_to_cn = {
        "bod": "身体",
        "mic": "微生物类",
        "dis": "疾病",
        "sym": "临床表现",
        "pro": "医疗程序",
        "equ": "医疗设备",
        "dru": "药物",
        "dep": "科室",
        "ite": "医学检验项目"
    }

    def _extract_chunk(self, tokens):
        chunks = set()
        start_idx, cur_idx = 0, 0
        while cur_idx < len(tokens):
            if tokens[cur_idx][0] == 'B':
                start_idx = cur_idx
                cur_idx += 1
                while cur_idx < len(tokens) and tokens[cur_idx][0] == 'I':
                    if tokens[cur_idx][2:] == tokens[start_idx][2:]:
                        cur_idx += 1
                    else:
                        break
                if cur_idx < len(tokens) and tokens[cur_idx][0] == 'E':
                    if tokens[cur_idx][2:] == tokens[start_idx][2:]:
                        chunks.add(
                            (tokens[cur_idx][2:], start_idx - 1, cur_idx))
                        cur_idx += 1
            elif tokens[cur_idx][0] == 'S':
                chunks.add((tokens[cur_idx][2:], cur_idx - 1, cur_idx))
                cur_idx += 1
            else:
                cur_idx += 1
        return list(chunks)

    def preprocess(self, infer_data):
        infer_data = [[x.lower() for x in text] for text in infer_data]
        data = self._tokenizer(infer_data,
                               max_length=self._max_seq_length,
                               padding=True,
                               is_split_into_words=True,
                               truncation=True,
                               return_position_ids=True,
                               return_attention_mask=True)

        encoded_inputs = {
            "input_ids": np.array(data["input_ids"], dtype="int64"),
            "token_type_ids": np.array(data["token_type_ids"], dtype="int64"),
            "position_ids": np.array(data["position_ids"], dtype="int64"),
            "attention_mask": np.array(data["attention_mask"], dtype="float32")
        }
        return encoded_inputs

    def postprocess(self, infer_data):
        tokens_oth = np.argmax(infer_data[0], axis=-1)
        tokens_sym = np.argmax(infer_data[1], axis=-1)
        entity = []
        for oth_ids, sym_ids in zip(tokens_oth, tokens_sym):
            token_oth = [self.label_list[0][x] for x in oth_ids]
            token_sym = [self.label_list[1][x] for x in sym_ids]
            chunks = self._extract_chunk(token_oth) \
                     + self._extract_chunk(token_sym)
            sub_entity = []
            for etype, sid, eid in chunks:
                sub_entity.append({
                    "type": self.en_to_cn[etype],
                    "start_id": sid,
                    "end_id": eid
                })
            entity.append(sub_entity)
        return {"entity": entity}

    def printer(self, result, input_data):
        result = result["entity"]
        for i, preds in enumerate(result):
            logger.info("input data: {}".format(input_data[i]))
            logger.info("detected entities:")
            for item in preds:
                logger.info("* entity: {}, type: {}, position: ({}, {})".format(
                    input_data[i][item["start_id"]:item["end_id"]],
                    item["type"], item["start_id"], item["end_id"]))
            logger.info("-----------------------------")


class SPOPredictor(EHealthPredictor):
    """ The predictor for the CMeIE dataset. """

    def predict(self, input_data: list):
        encoded_inputs = self.preprocess(input_data)
        lengths = encoded_inputs["attention_mask"].sum(axis=-1)
        infer_result = self.infer_batch(encoded_inputs)
        result = self.postprocess(infer_result, lengths)
        self.printer(result, input_data)
        return result

    def preprocess(self, infer_data):
        infer_data = [[x.lower() for x in text] for text in infer_data]
        data = self._tokenizer(infer_data,
                               max_length=self._max_seq_length,
                               padding=True,
                               is_split_into_words=True,
                               truncation=True,
                               return_position_ids=True,
                               return_attention_mask=True)

        encoded_inputs = {
            "input_ids": np.array(data["input_ids"], dtype="int64"),
            "token_type_ids": np.array(data["token_type_ids"], dtype="int64"),
            "position_ids": np.array(data["position_ids"], dtype="int64"),
            "attention_mask": np.array(data["attention_mask"], dtype="float32")
        }
        return encoded_inputs

    def postprocess(self, infer_data, lengths):
        ent_logits = np.array(infer_data[0])
        spo_logits = np.array(infer_data[1])
        ent_pred_list = []
        ent_idxs_list = []
        for idx, ent_pred in enumerate(ent_logits):
            seq_len = lengths[idx] - 2
            start = np.where(ent_pred[:, 0] > 0.5)[0]
            end = np.where(ent_pred[:, 1] > 0.5)[0]
            ent_pred = []
            ent_idxs = {}
            for x in start:
                y = end[end >= x]
                if (x == 0) or (x > seq_len):
                    continue
                if len(y) > 0:
                    y = y[0]
                    if y > seq_len:
                        continue
                    ent_idxs[x] = (x - 1, y - 1)
                    ent_pred.append((x - 1, y - 1))
            ent_pred_list.append(ent_pred)
            ent_idxs_list.append(ent_idxs)

        spo_preds = spo_logits > 0
        spo_pred_list = [[] for _ in range(len(spo_preds))]
        idxs, preds, subs, objs = np.nonzero(spo_preds)
        for idx, p_id, s_id, o_id in zip(idxs, preds, subs, objs):
            obj = ent_idxs_list[idx].get(o_id, None)
            if obj is None:
                continue
            sub = ent_idxs_list[idx].get(s_id, None)
            if sub is None:
                continue
            spo_pred_list[idx].append((tuple(sub), p_id, tuple(obj)))

        return {"entity": ent_pred_list, "spo": spo_pred_list}

    def printer(self, result, input_data):
        ent_pred_list, spo_pred_list = result["entity"], result["spo"]
        for i, (ent, rel) in enumerate(zip(ent_pred_list, spo_pred_list)):
            logger.info("input data: {}".format(input_data[i]))
            logger.info("detected entities and relations:")
            for sid, eid in ent:
                logger.info("* entity: {}, position: ({}, {})".format(
                    input_data[i][sid:eid + 1], sid, eid))
            for s, p, o in rel:
                logger.info("+ spo: ({}, {}, {})".format(
                    input_data[i][s[0]:s[1] + 1], self.label_list[p],
                    input_data[i][o[0]:o[1] + 1]))
            logger.info("-----------------------------")
