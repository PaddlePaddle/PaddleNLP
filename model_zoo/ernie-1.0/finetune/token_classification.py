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


def tokenize_and_align_labels(example,
                              tokenizer,
                              no_entity_id,
                              max_seq_len=512):
    if 'labels' in example:
        labels = example['labels']
        example = example['tokens']
        tokenized_input = tokenizer(example,
                                    is_split_into_words=True,
                                    max_seq_len=max_seq_len,
                                    return_length=False)

        # -2 for [CLS] and [SEP]
        if len(tokenized_input['input_ids']) - 2 < len(labels):
            labels = labels[:len(tokenized_input['input_ids']) - 2]
        tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
        tokenized_input['labels'] += [no_entity_id] * (
            len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    else:
        if example['tokens'] == []:
            tokenized_input = {
                'labels': [],
                'input_ids': [],
                'token_type_ids': [],
            }
            return tokenized_input
        tokenized_input = tokenizer(
            example['tokens'],
            max_seq_len=max_seq_len,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_length=False)
        label_ids = example['ner_tags']
        if len(tokenized_input['input_ids']) - 2 < len(label_ids):
            label_ids = label_ids[:len(tokenized_input['input_ids']) - 2]
        label_ids = [no_entity_id] + label_ids + [no_entity_id]

        label_ids += [no_entity_id
                      ] * (len(tokenized_input['input_ids']) - len(label_ids))
        tokenized_input["labels"] = label_ids
    return tokenized_input


def ner_trans_fn(example, tokenizer, args):
    return tokenize_and_align_labels(example,
                                     tokenizer=tokenizer,
                                     no_entity_id=args.no_entity_id,
                                     max_seq_len=args.max_seq_length)
