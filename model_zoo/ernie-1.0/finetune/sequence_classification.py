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

import numpy as np


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    is_test = True
    if 'label' in example.keys():
        is_test = False

    if "text_b" in example.keys():
        text = example["text_a"]
        text_pair = example["text_b"]
    else:
        text = example["text"]
        text_pair = None

    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair,
                               max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if is_test:
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
        }
    else:
        # label = np.array([example["label"]], dtype="int64")
        label = int(example["label"])
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "labels": label
        }


# Data pre-process function for clue benchmark datatset
def convert_clue(example,
                 label_list,
                 tokenizer=None,
                 max_seq_length=512,
                 **kwargs):
    """convert a glue example into necessary features"""
    is_test = False
    if 'label' not in example.keys():
        is_test = True

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        example['label'] = int(
            example["label"]) if label_dtype != "float32" else float(
                example["label"])
        label = example['label']
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
        assert text[pronoun_idx:(
            pronoun_idx +
            len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx +
                               len(query))] == query, "query: {}".format(query)
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

    if tokenizer is None:
        return example
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(example['sentence1'],
                            text_pair=example['sentence2'],
                            max_seq_len=max_seq_length)

    if not is_test:
        return {
            "input_ids": example['input_ids'],
            "token_type_ids": example['token_type_ids'],
            "labels": label
        }
    else:
        return {
            "input_ids": example['input_ids'],
            "token_type_ids": example['token_type_ids']
        }


def seq_trans_fn(example, tokenizer, args):
    return convert_example(
        example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )


def clue_trans_fn(example, tokenizer, args):
    return convert_clue(example,
                        tokenizer=tokenizer,
                        label_list=args.label_list,
                        max_seq_length=args.max_seq_length)
