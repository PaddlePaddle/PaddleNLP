# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from pypinyin import lazy_pinyin, Style
import paddle


def read_train_ds(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            words, labels = line.strip('\n').split('\t')
            pinyins = lazy_pinyin(
                words, style=Style.TONE3, neutral_tone_with_five=True)
            yield {'words': words, 'pinyins': pinyins, 'labels': labels}


def read_test_ds(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            ids, words = line.strip('\n').split('\t')
            pinyins = lazy_pinyin(
                words, style=Style.TONE3, neutral_tone_with_five=True)
            yield {'words': words, 'pinyins': pinyins}


def convert_example(example,
                    tokenizer,
                    pinyin_vocab,
                    max_seq_length=128,
                    ignore_label=-1,
                    is_test=False):
    words = tokenizer.tokenize(text=example["words"])
    if len(words) > max_seq_length - 2:
        words = words[:max_seq_length - 2]
    length = len(words)
    words = ['[CLS]'] + words + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(words)
    token_type_ids = [0] * len(input_ids)

    # use pad token in pinyin emb to map word emb [CLS], [SEP]
    pinyins = example["pinyins"]
    pinyin_ids = [0]
    # align pinyin and chinese char
    pinyin_offset = 0
    for i, word in enumerate(words[1:-1]):
        pinyin = '[UNK]' if word != '[PAD]' else '[PAD]'
        if len(word) == 1 and is_chinese_char(ord(word)):
            while pinyin_offset < len(pinyins):
                current_pinyin = pinyins[pinyin_offset][:-1]
                pinyin_offset += 1
                if current_pinyin in pinyin_vocab:
                    pinyin = current_pinyin
                    break
        pinyin_ids.append(pinyin_vocab[pinyin])

    pinyin_ids.append(0)
    assert len(input_ids) == len(
        pinyin_ids), "length of input_ids must be equal to length of pinyin_ids"

    if not is_test:
        correction_labels = tokenizer.tokenize(text=example["labels"])
        if len(correction_labels) > max_seq_length - 2:
            correction_labels = correction_labels[:max_seq_length - 2]
        correction_labels = tokenizer.convert_tokens_to_ids(correction_labels)
        correction_labels = [ignore_label] + correction_labels + [ignore_label]

        detection_labels = []
        for input_id, label in zip(input_ids[1:-1], correction_labels[1:-1]):
            detection_label = 0 if input_id == label else 1
            detection_labels += [detection_label]
        detection_labels = [ignore_label] + detection_labels + [ignore_label]
        return input_ids, token_type_ids, pinyin_ids, detection_labels, correction_labels
    else:
        return input_ids, token_type_ids, pinyin_ids, length


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


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False
