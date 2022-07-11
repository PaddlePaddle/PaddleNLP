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

import re
import os
import json

from paddlenlp.transformers import AutoTokenizer

from utils import Preprocessor


def clean_text(text):
    text = re.sub(u"\u3000", " ", text)
    return text


def do_convert(input_file, target_file, label_map, preprocessor):
    with open(input_file, 'r', encoding='utf-8') as f:
        id = 0
        outputs = []
        for line in f:
            output = {}
            json_line = json.loads(line)
            output['id'] = id
            output['text'] = clean_text(json_line['text'])
            spo_list = json_line['spo_list']
            for spo in spo_list:
                subject_text = clean_text(spo['subject'])
                if len(subject_text) == 0:
                    continue
                predicate_text = clean_text(spo['predicate'])
                subject_type = spo['subject_type']
                entity = {'text': subject_text, 'type': subject_type}
                output['entity_list'] = [entity]
                for spo_object in spo['object'].keys():
                    object_text = clean_text(spo['object'][spo_object])
                    if len(object_text) == 0:
                        continue
                    object_type = spo['object_type'][spo_object]
                    entity = {'text': object_text, 'type': object_type}
                    output['entity_list'].append(entity)
                    if predicate_text in label_map.keys():
                        # simple relation
                        relation = {
                            'subject': subject_text,
                            'object': object_text,
                            'predicate': predicate_text
                        }
                    else:
                        # complex relation
                        relation = {
                            'subject': subject_text,
                            'object': object_text,
                            'predicate': predicate_text + '_' + spo_object
                        }
                    output.setdefault('relation_list', []).append(relation)
            outputs.append(output)
            id += 1

    # Add char span
    outputs, _ = preprocessor.add_char_span(outputs, False)

    # Add token span
    outputs = preprocessor.add_tok_span(outputs)

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(outputs, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    data_home = "./data"
    input_file_list = ["duie_train.json", "duie_dev.json"]
    target_file_list = ["train_data.json", "valid_data.json"]
    rel2id_path = os.path.join(data_home, 'rel2id.json')
    with open(rel2id_path, 'r', encoding='utf8') as fp:
        label_map = json.load(fp)

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh",
                                              use_faster=True)

    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer(text,
                                                   return_token_type_ids=None,
                                                   return_offsets_mapping=True,
                                                   add_special_tokens=False)[
                                                       "offset_mapping"]

    preprocessor = Preprocessor(
        tokenize_func=tokenize,
        get_tok2char_span_map_func=get_tok2char_span_map)

    for fi, ft in zip(input_file_list, target_file_list):
        input_file_path = os.path.join(data_home, fi)
        target_file_path = os.path.join(data_home, ft)
        do_convert(input_file_path, target_file_path, label_map, preprocessor)
