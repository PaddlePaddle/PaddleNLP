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
import logging
import json
import numpy as np
from numpy import array

from paddle_serving_server.web_service import WebService, Op

_LOGGER = logging.getLogger()


class ErnieHealthTokenClsOp(Op):
    """Op of token classification task on ERNIE-Health"""

    def init_op(self):
        """Initialize the op configuration"""
        from paddlenlp.transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'ernie-health-chinese', use_faster=True)
        self.label_names = [[
            'B-bod', 'I-bod', 'E-bod', 'S-bod', 'B-dis', 'I-dis', 'E-dis',
            'S-dis', 'B-pro', 'I-pro', 'E-pro', 'S-pro', 'B-dru', 'I-dru',
            'E-dru', 'S-dru', 'B-ite', 'I-ite', 'E-ite', 'S-ite', 'B-mic',
            'I-mic', 'E-mic', 'S-mic', 'B-equ', 'I-equ', 'E-equ', 'S-equ',
            'B-dep', 'I-dep', 'E-dep', 'S-dep', 'O'
        ], ['B-sym', 'I-sym', 'E-sym', 'S-sym', 'O']]
        # Check the serving_server_conf.prototxt file in serving_server, and 
        # change the fetch_names list to that/those found in the file.
        self.fetch_names = ["linear_146.tmp_1", "linear_147.tmp_1"]

    def get_input_data(self, input_dicts):
        """Extract the input data uploaded by client"""
        (_, input_dict), = input_dicts.items()
        data = input_dict["tokens"]

        if isinstance(data, str) and "array(" in data:
            data = eval(data)
            data = data.tolist()
        else:
            _LOGGER.error("input value {} is not supported.".format(data))
        return data

    def preprocess(self, input_dicts, data_id, log_id):
        """
        Args:
            input_dicts: input data to be preprocessed
            data_id: inner unique id, increase auto
            log_id: global unique id for RTT, 0 default
        Return:
            output_data: data for process stage
            is_skip_process: skip process stage or not, False default
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception. 
            prod_errinfo: "" default
        """
        # convert input format
        data = self.get_input_data(input_dicts)

        # tokenizer + pad
        data = self.tokenizer(
            data,
            max_length=128,
            padding=True,
            truncation=True,
            return_position_ids=True,
            return_attention_mask=True)

        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]
        return {
            "input_ids": np.array(
                input_ids, dtype="int64"),
            "token_type_ids": np.array(
                token_type_ids, dtype="int64"),
            "position_ids": np.array(
                position_ids, dtype="int64"),
            "attention_mask": np.array(
                attention_mask, dtype="float")
        }, False, None, ""

    def _extract_tokens(self, preds, label_names):
        tokens = [label_names[x] for x in preds]
        entity_set = set()

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
                if tokens[cur_idx][0] == 'E':
                    if tokens[cur_idx][2:] == tokens[start_idx][2:]:
                        etype = tokens[cur_idx][2:]
                        entity_set.add((etype, start_idx, cur_idx))
                        cur_idx += 1
            elif tokens[cur_idx][0] == 'S':
                etype = tokens[cur_idx][2:]
                entity_set.add((etype, cur_idx, cur_idx))
                cur_idx += 1
            else:
                cur_idx += 1
        return entity_set

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        """
        In postprocess stage, assemble data for next op or output.
        Args:
            input_data: data returned in preprocess stage, dict(for single predict) or list(for batch predict)
            fetch_data: data returned in process stage, dict(for single predict) or list(for batch predict)
            data_id: inner unique id, increase auto
            log_id: logid, 0 default
        Returns: 
            fetch_dict: fetch result must be dict type.
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception.
            prod_errinfo: "" default
        """
        input_data = self.get_input_data(input_dicts)
        sym_tokens = fetch_dict[self.fetch_names[0]].argmax(axis=-1).tolist()
        oth_tokens = fetch_dict[self.fetch_names[1]].argmax(axis=-1).tolist()

        entity_list = []
        for idx, (o_tokens, s_tokens) in enumerate(zip(oth_tokens, sym_tokens)):
            text = input_data[idx]
            sym_entity = self._extract_tokens(s_tokens, self.label_names[1])
            oth_entity = self._extract_tokens(o_tokens, self.label_names[0])
            sub_entity = []
            for etype, sid, eid in list(sym_entity | oth_entity):
                sub_entity.append({'type': etype, 'position': (sid, eid)})
            entity_list.append(sub_entity)

        out_dict = {
            "value": json.dumps(entity_list),
            "sym_predicts": json.dumps(sym_tokens),
            "oth_predicts": json.dumps(oth_tokens)
        }
        return out_dict, None, ""


class ErnieHealthTokenClsService(WebService):
    """ErnieHealthTokenClsService"""

    def get_pipeline_response(self, read_op):
        return ErnieHealthTokenClsOp(name="token_cls", input_ops=[read_op])


if __name__ == "__main__":
    ocr_service = ErnieHealthTokenClsService(name="token_cls")
    ocr_service.prepare_pipeline_config("token_cls_config.yml")
    ocr_service.run_service()
