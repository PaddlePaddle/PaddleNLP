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
    words = list(source)
    if len(words) > max_seq_length - 2:
        words = words[:max_seq_length - 2]
    length = len(words)
    words = ['[CLS]'] + words + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(words)
    token_type_ids = [0] * len(input_ids)

    # Use pad token in pinyin emb to map word emb [CLS], [SEP]
    pinyins = lazy_pinyin(source,
                          style=Style.TONE3,
                          neutral_tone_with_five=True)
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
        correction_labels = list(target)
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


def parse_decode(words, corr_preds, det_preds, lengths, tokenizer,
                 max_seq_length):
    UNK = tokenizer.unk_token
    UNK_id = tokenizer.convert_tokens_to_ids(UNK)

    corr_pred = corr_preds[1:1 + lengths].tolist()
    det_pred = det_preds[1:1 + lengths].tolist()
    words = list(words)
    rest_words = []
    if len(words) > max_seq_length - 2:
        rest_words = words[max_seq_length - 2:]
        words = words[:max_seq_length - 2]
    pred_result = ""
    for j, word in enumerate(words):
        candidates = tokenizer.convert_ids_to_tokens(
            corr_pred[j] if corr_pred[j] < tokenizer.vocab_size else UNK_id)
        word_icc = is_chinese_char(ord(word))
        cand_icc = is_chinese_char(
            ord(candidates)) if len(candidates) == 1 else False
        if not word_icc or det_pred[j] == 0\
            or candidates in [UNK, '[PAD]']\
            or (word_icc and not cand_icc):
            pred_result += word
        else:
            pred_result += candidates.lstrip("##")
    pred_result += ''.join(rest_words)
    return pred_result
