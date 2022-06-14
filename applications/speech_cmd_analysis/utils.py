# coding=utf-8
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

import json
import time
import math
import random
import numpy as np
from tqdm import tqdm

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode

import paddle

MODEL_MAP = {
    "uie-base": {
        "encoding_model": "ernie-3.0-base-zh",
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json"
        }
    },
    "uie-tiny": {
        "encoding_model": "ernie-3.0-medium-zh",
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_config.json"
        }
    },
}


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class ASRError(Exception):
    pass


def mandarin_asr_api(api_key, secret_key, audio_file, audio_format='wav'):
    """ Mandarin ASR

    Args:
        audio_file (str):
            Audio file of Mandarin with sampling rate 16000.
        audio_format (str):
            The file extension of audio_file, 'wav' by default.

    Please refer to https://github.com/Baidu-AIP/speech-demo for more demos.
    """
    # Configurations.
    TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
    ASR_URL = 'http://vop.baidu.com/server_api'
    SCOPE = 'audio_voice_assistant_get'
    API_KEY = api_key
    SECRET_KEY = secret_key

    # Fetch tokens from TOKEN_URL.
    post_data = urlencode({
        'grant_type': 'client_credentials',
        'client_id': API_KEY,
        'client_secret': SECRET_KEY
    }).encode('utf-8')

    request = Request(TOKEN_URL, post_data)
    try:
        result_str = urlopen(request).read()
    except URLError as error:
        print('token http response http code : ' + str(error.code))
        result_str = err.read()
    result_str = result_str.decode()

    result = json.loads(result_str)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if SCOPE and (not SCOPE in result['scope'].split(' ')):
            raise ASRError('scope is not correct!')
        token = result['access_token']
    else:
        raise ASRError('MAYBE API_KEY or SECRET_KEY not correct: ' +
                       'access_token or scope not found in token response')

    # Fetch results by ASR api.
    with open(audio_file, 'rb') as speech_file:
        speech_data = speech_file.read()
    length = len(speech_data)
    if length == 0:
        raise ASRError('file %s length read 0 bytes' % audio_file)
    params_query = urlencode({'cuid': 'ASR', 'token': token, 'dev_pid': 1537})
    headers = {
        'Content-Type': 'audio/%s; rate=16000' % audio_format,
        'Content-Length': length
    }

    url = ASR_URL + '?' + params_query
    request = Request(url, speech_data, headers)
    try:
        begin = time.time()
        result_str = urlopen(request).read()
        print('Request time cost %f' % (time.time() - begin))
    except URLError as error:
        print('asr http response http code : ' + str(error.code))
        result_str = error.read()
    result_str = str(result_str, 'utf-8')
    result = json.loads(result_str)

    return result['result'][0]


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, token_type_ids, att_mask, pos_ids, start_ids, end_ids = batch
        start_prob, end_prob = model(input_ids, token_type_ids, att_mask,
                                     pos_ids)
        start_ids = paddle.cast(start_ids, 'float32')
        end_ids = paddle.cast(end_ids, 'float32')
        num_correct, num_infer, num_label = metric.compute(
            start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    return precision, recall, f1


def convert_example(example, tokenizer, max_seq_len):
    """
    example: {
        title
        prompt
        content
        result_list
    }
    """
    encoded_inputs = tokenizer(text=[example["prompt"]],
                               text_pair=[example["content"]],
                               stride=len(example["prompt"]),
                               truncation=True,
                               max_seq_len=max_seq_len,
                               pad_to_max_seq_len=True,
                               return_attention_mask=True,
                               return_position_ids=True,
                               return_dict=False)
    encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
    bias = 0
    for index in range(len(offset_mapping)):
        if index == 0:
            continue
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = index
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0 for x in range(max_seq_len)]
    end_ids = [0 for x in range(max_seq_len)]
    for item in example["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    tokenized_output = [
        encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
        encoded_inputs["position_ids"], encoded_inputs["attention_mask"],
        start_ids, end_ids
    ]
    tokenized_output = [np.array(x, dtype="int64") for x in tokenized_output]
    return tuple(tokenized_output)


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def reader(data_path, max_seq_len=512):
    """
    read json
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line['content']
            prompt = json_line['prompt']
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError(
                    "The value of max_seq_len is too small, please set a larger value"
                )
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line['result_list']
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result['start'] + 1 <= max_content_len < result[
                                'end']:
                            max_content_len = result['start']
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]['end'] <= max_content_len:
                            if result_list[0]['end'] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [
                                    result for result in result_list
                                ]
                                break
                        else:
                            break

                    json_line = {
                        'content': cur_content,
                        'result_list': cur_result_list,
                        'prompt': prompt
                    }
                    json_lines.append(json_line)

                    for result in result_list:
                        if result['end'] <= 0:
                            break
                        result['start'] -= max_content_len
                        result['end'] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {
                            'content': res_content,
                            'result_list': result_list,
                            'prompt': prompt
                        }
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


def add_negative_example(examples, texts, prompts, label_set, negative_ratio):
    with tqdm(total=len(prompts)) as pbar:
        for i, prompt in enumerate(prompts):
            negtive_sample = []
            redundants_list = list(set(label_set) ^ set(prompt))
            redundants_list.sort()

            if len(examples[i]) == 0:
                continue
            else:
                actual_ratio = math.ceil(
                    len(redundants_list) / len(examples[i]))

            if actual_ratio <= negative_ratio:
                idxs = [k for k in range(len(redundants_list))]
            else:
                idxs = random.sample(range(0, len(redundants_list)),
                                     negative_ratio * len(examples[i]))

            for idx in idxs:
                negtive_result = {
                    "content": texts[i],
                    "result_list": [],
                    "prompt": redundants_list[idx]
                }
                negtive_sample.append(negtive_result)
            examples[i].extend(negtive_sample)
            pbar.update(1)
    return examples


def construct_relation_prompt_set(entity_name_set, predicate_set):
    relation_prompt_set = set()
    for entity_name in entity_name_set:
        for predicate in predicate_set:
            # The relation prompt is constructed as follows:
            # subject + "的" + predicate
            relation_prompt = entity_name + "的" + predicate
            relation_prompt_set.add(relation_prompt)
    return sorted(list(relation_prompt_set))


def convert_ext_examples(raw_examples, negative_ratio):
    texts = []
    entity_examples = []
    relation_examples = []
    entity_prompts = []
    relation_prompts = []
    entity_label_set = []
    entity_name_set = []
    predicate_set = []

    print(f"Converting doccano data...")
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            entity_id = 0
            if "data" in items.keys():
                text = items["data"]
                entities = []
                for item in items["label"]:
                    entity = {
                        "id": entity_id,
                        "start_offset": item[0],
                        "end_offset": item[1],
                        "label": item[2]
                    }
                    entities.append(entity)
                    entity_id += 1
                relations = []
            else:
                text, relations, entities = items["text"], items[
                    "relations"], items["entities"]
            texts.append(text)

            entity_example = []
            entity_prompt = []
            entity_example_map = {}
            entity_map = {}  # id to entity name
            for entity in entities:
                entity_name = text[entity["start_offset"]:entity["end_offset"]]
                entity_map[entity["id"]] = {
                    "name": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }

                entity_label = entity["label"]
                result = {
                    "text": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }
                if entity_label not in entity_example_map.keys():
                    entity_example_map[entity_label] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": entity_label
                    }
                else:
                    entity_example_map[entity_label]["result_list"].append(
                        result)

                if entity_label not in entity_label_set:
                    entity_label_set.append(entity_label)
                if entity_name not in entity_name_set:
                    entity_name_set.append(entity_name)
                entity_prompt.append(entity_label)

            for v in entity_example_map.values():
                entity_example.append(v)

            entity_examples.append(entity_example)
            entity_prompts.append(entity_prompt)

            relation_example = []
            relation_prompt = []
            relation_example_map = {}
            for relation in relations:
                predicate = relation["type"]
                subject_id = relation["from_id"]
                object_id = relation["to_id"]
                # The relation prompt is constructed as follows:
                # subject + "的" + predicate
                prompt = entity_map[subject_id]["name"] + "的" + predicate
                result = {
                    "text": entity_map[object_id]["name"],
                    "start": entity_map[object_id]["start"],
                    "end": entity_map[object_id]["end"]
                }
                if prompt not in relation_example_map.keys():
                    relation_example_map[prompt] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": prompt
                    }
                else:
                    relation_example_map[prompt]["result_list"].append(result)

                if predicate not in predicate_set:
                    predicate_set.append(predicate)
                relation_prompt.append(prompt)

            for v in relation_example_map.values():
                relation_example.append(v)

            relation_examples.append(relation_example)
            relation_prompts.append(relation_prompt)
            pbar.update(1)

    print(f"Adding negative samples for first stage prompt...")
    entity_examples = add_negative_example(entity_examples, texts,
                                           entity_prompts, entity_label_set,
                                           negative_ratio)
    if len(predicate_set) != 0:
        print(f"Constructing relation prompts...")
        relation_prompt_set = construct_relation_prompt_set(
            entity_name_set, predicate_set)

        print(f"Adding negative samples for second stage prompt...")
        relation_examples = add_negative_example(relation_examples, texts,
                                                 relation_prompts,
                                                 relation_prompt_set,
                                                 negative_ratio)
    return entity_examples, relation_examples


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)
