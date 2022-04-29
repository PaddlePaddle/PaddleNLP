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
def evaluate(model, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    num_correct = 0
    num_infer = 0
    num_label = 0
    for batch in data_loader:
        input_ids, token_type_ids, att_mask, pos_ids, start_ids, end_ids = batch
        start_prob, end_prob = model(input_ids, token_type_ids, att_mask,
                                     pos_ids)
        start_ids = paddle.cast(start_ids, 'float32')
        end_ids = paddle.cast(end_ids, 'float32')
        res = get_metric(start_prob, end_prob, start_ids, end_ids)
        num_correct += res[0]
        num_infer += res[1]
        num_label += res[2]
    precision, recall, f1 = get_f1(num_correct, num_infer, num_label)
    model.train()
    return precision, recall, f1


def get_eval(tokenizer, step, data_loader, model, name):
    """
    eval test set
    """
    num_correct = 0
    num_infer = 0
    num_label = 0
    fw_gold = open(
        'output/prediction/' + name + '-gold.' + str(step),
        'w+',
        encoding='utf8')
    fw_pred = open(
        'output/prediction/' + name + '-pred.' + str(step),
        'w+',
        encoding='utf8')
    for [input_ids, token_type_ids, att_mask, pos_ids, start_ids,
         end_ids] in data_loader():
        start_prob, end_prob = model(input_ids, token_type_ids, att_mask,
                                     pos_ids)
        start_ids = paddle.cast(start_ids, 'float32')
        end_ids = paddle.cast(end_ids, 'float32')
        res = get_metric(start_prob, end_prob, start_ids, end_ids)
        num_correct += res[0]
        num_infer += res[1]
        num_label += res[2]
        get_result(tokenizer,
                   input_ids.tolist(),
                   start_ids.tolist(), end_ids.tolist(), fw_gold)
        get_result(tokenizer,
                   input_ids.tolist(),
                   start_prob.tolist(), end_prob.tolist(), fw_pred)
    fw_gold.close()
    fw_pred.close()
    res = get_f1(num_correct, num_infer, num_label)
    print('--%s --F1 %.4f --P %.4f (%i / %i) --R %.4f (%i / %i)' %
          (name, res[2], res[0], num_correct, num_infer, res[1], num_correct,
           num_label))
    return res[2]


def get_metric(start_prob, end_prob, start_ids, end_ids):
    """
    get_metric
    """
    pred_start_ids = get_bool_ids_greater_than(start_prob)
    pred_end_ids = get_bool_ids_greater_than(end_prob)
    gold_start_ids = get_bool_ids_greater_than(start_ids.tolist())
    gold_end_ids = get_bool_ids_greater_than(end_ids.tolist())

    num_correct = 0
    num_infer = 0
    num_label = 0
    for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
            pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids):
        [_correct, _infer, _label] = eval_span(
            predict_start_ids, predict_end_ids, label_start_ids, label_end_ids)
        num_correct += _correct
        num_infer += _infer
        num_label += _label
    return num_correct, num_infer, num_label


def get_f1(num_correct, num_infer, num_label):
    """
    get p r f1
    input: 10, 15, 20
    output: (0.6666666666666666, 0.5, 0.5714285714285715)
    """
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def get_result(tokenizer, src_ids, start_prob, end_prob, fw):
    """
    get_result
    """
    start_ids_list = get_bool_ids_greater_than(start_prob)
    end_ids_list = get_bool_ids_greater_than(end_prob)
    for start_ids, end_ids, ids in zip(start_ids_list, end_ids_list, src_ids):
        for i in reversed(range(len(ids))):
            if ids[i] != 0:
                ids = ids[:i]
                break
        span_list = get_span(start_ids, end_ids)
        src_words = " ".join(tokenizer.convert_ids_to_tokens(ids))
        span_words = [
            " ".join(tokenizer.convert_ids_to_tokens(ids[s[0]:(s[1] + 1)]))
            for s in span_list
        ]
        fw.writelines(src_words + "\n")
        fw.writelines(json.dumps(span_words, ensure_ascii=False) + "\n\n")
    return None


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    get idx of the last dim in prob arraies, which is greater than a limitation
    input: [[0.1, 0.1, 0.2, 0.5, 0.1, 0.3], [0.7, 0.6, 0.1, 0.1, 0.1, 0.1]]
        0.4
    output: [[3], [0, 1]]
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span(start_ids, end_ids, with_prob=False):
    """
    every id can only be used once
    get span set from position start and end list
    input: [1, 2, 10] [4, 12]
    output: set((2, 4), (10, 12))
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
                end_pointer += 1
                continue
        else:
            if start_ids[start_pointer] == end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer] < end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer] > end_ids[end_pointer]:
                end_pointer += 1
                continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def eval_span(predict_start_ids, predict_end_ids, label_start_ids,
              label_end_ids):
    """
    evaluate position extraction (start, end)
    return num_correct, num_infer, num_label
    input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
    output: (1, 2, 2)
    """
    pred_set = get_span(predict_start_ids, predict_end_ids)
    label_set = get_span(label_start_ids, label_end_ids)
    num_correct = len(pred_set & label_set)
    num_infer = len(pred_set)
    num_label = len(label_set)
    return (num_correct, num_infer, num_label)


def convert_example(example, tokenizer, max_seq_len):
    """
    example: {
        title
        prompt
        content
        result_list
    }
    """
    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        stride=len(example["prompt"]),
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


def reader(data_path):
    """
    read json
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            yield json_line


def save_examples(examples, save_path, idxs):
    with open(save_path, "w", encoding="utf-8") as f:
        for idx in idxs:
            for example in examples[idx]:
                line = json.dumps(example, ensure_ascii=False) + "\n"
                f.write(line)


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
                idxs = random.sample(
                    range(0, len(redundants_list)),
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


def construct_relation_label_set(entity_name_set, predicate_set):
    relation_label_set = set()
    for entity_name in entity_name_set:
        for predicate in predicate_set:
            relation_label = entity_name + "的" + predicate
            relation_label_set.add(relation_label)
    return sorted(list(relation_label_set))


def convert_data_examples(raw_examples, negative_ratio):
    texts = []
    entity_examples = []
    relation_examples = []
    entity_prompts = []
    relation_prompts = []
    entity_label_set = []
    entity_name_set = []
    predicate_set = []

    print(f"Converting data...")
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
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
                relation_label = entity_map[subject_id][
                    "name"] + "的" + predicate
                result = {
                    "text": entity_map[object_id]["name"],
                    "start": entity_map[object_id]["start"],
                    "end": entity_map[object_id]["end"]
                }
                if relation_label not in relation_example_map.keys():
                    relation_example_map[relation_label] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": relation_label
                    }
                else:
                    relation_example_map[relation_label]["result_list"].append(
                        result)

                if predicate not in predicate_set:
                    predicate_set.append(predicate)
                relation_prompt.append(relation_label)

            for v in relation_example_map.values():
                relation_example.append(v)

            relation_examples.append(relation_example)
            relation_prompts.append(relation_prompt)
            pbar.update(1)

    print(f"Adding negative samples for first stage prompt...")
    entity_examples = add_negative_example(entity_examples, texts,
                                           entity_prompts, entity_label_set,
                                           negative_ratio)

    print(f"Constructing relation labels...")
    relation_label_set = construct_relation_label_set(entity_name_set,
                                                      predicate_set)

    print(f"Adding negative samples for second stage prompt...")
    relation_examples = add_negative_example(relation_examples, texts,
                                             relation_prompts,
                                             relation_label_set, negative_ratio)
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
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
