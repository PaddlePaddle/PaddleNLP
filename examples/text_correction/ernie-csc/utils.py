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

from paddlenlp.transformers import is_chinese_char


def read_train_ds(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            source, target = line.strip('\n').split('\t')[0:2]
            yield {'source': source, 'target': target}


def read_test_ds(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            ids, words = line.strip('\n').split('\t')[0:2]
            yield {'source': words}


def convert_example(example,
                    tokenizer,
                    pinyin_vocab,
                    max_seq_length=128,
                    ignore_label=-1,
                    is_test=False):
    source = example["source"]
    words = tokenizer.tokenize(text=source)
    if len(words) > max_seq_length - 2:
        words = words[:max_seq_length - 2]
    length = len(words)
    words = ['[CLS]'] + words + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(words)
    token_type_ids = [0] * len(input_ids)

    # Use pad token in pinyin emb to map word emb [CLS], [SEP]
    pinyins = lazy_pinyin(
        source, style=Style.TONE3, neutral_tone_with_five=True)

    pinyin_ids = [0]
    # Align pinyin and chinese char
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
        target = example["target"]
        correction_labels = tokenizer.tokenize(text=target)
        if len(correction_labels) > max_seq_length - 2:
            correction_labels = correction_labels[:max_seq_length - 2]
        correction_labels = tokenizer.convert_tokens_to_ids(correction_labels)
        correction_labels = [ignore_label] + correction_labels + [ignore_label]

        detection_labels = []
        for input_id, label in zip(input_ids[1:-1], correction_labels[1:-1]):
            detection_label = 0 if input_id == label else 1
            detection_labels += [detection_label]
        detection_labels = [ignore_label] + detection_labels + [ignore_label]
        return input_ids, token_type_ids, pinyin_ids, detection_labels, correction_labels, length
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


def parse_decode(words, corr_preds, det_preds, lengths, tokenizer,
                 max_seq_length):
    UNK = tokenizer.unk_token
    UNK_id = tokenizer.convert_tokens_to_ids(UNK)
    tokens = tokenizer.tokenize(words)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    corr_pred = corr_preds[1:1 + lengths].tolist()
    det_pred = det_preds[1:1 + lengths].tolist()
    words = list(words)
    if len(words) > max_seq_length - 2:
        words = words[:max_seq_length - 2]

    assert len(tokens) == len(
        corr_pred
    ), "The number of tokens should be equal to the number of labels {}: {}: {}".format(
        len(tokens), len(corr_pred), tokens)
    pred_result = ""

    align_offset = 0
    # Need to be aligned
    if len(words) != len(tokens):
        first_unk_flag = True
        for j, word in enumerate(words):
            if word.isspace():
                tokens.insert(j + 1, word)
                corr_pred.insert(j + 1, UNK_id)
                det_pred.insert(j + 1, 0)  # No error
            elif tokens[j] != word:
                if tokenizer.convert_tokens_to_ids(word) == UNK_id:
                    if first_unk_flag:
                        first_unk_flag = False
                        corr_pred[j] = UNK_id
                        det_pred[j] = 0
                    else:
                        tokens.insert(j, UNK)
                        corr_pred.insert(j, UNK_id)
                        det_pred.insert(j, 0)  # No error
                    continue
                elif tokens[j] == UNK:
                    # Remove rest unk
                    k = 0
                    while k + j < len(tokens) and tokens[k + j] == UNK:
                        k += 1
                    tokens = tokens[:j] + tokens[j + k:]
                    corr_pred = corr_pred[:j] + corr_pred[j + k:]
                    det_pred = det_pred[:j] + det_pred[j + k:]
                else:
                    # Maybe English, number, or suffix
                    token = tokens[j].lstrip("##")
                    corr_pred = corr_pred[:j] + [UNK_id] * len(
                        token) + corr_pred[j + 1:]
                    det_pred = det_pred[:j] + [0] * len(token) + det_pred[j +
                                                                          1:]
                    tokens = tokens[:j] + list(token) + tokens[j + 1:]
            first_unk_flag = True

    for j, word in enumerate(words):
        candidates = tokenizer.convert_ids_to_tokens(corr_pred[j])
        if det_pred[j] == 0 or candidates == UNK or candidates == '[PAD]':
            pred_result += word
        else:
            pred_result += candidates.lstrip("##")

    return pred_result
