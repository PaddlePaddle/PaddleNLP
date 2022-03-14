'''
@author: zhangxinrui
@name: dataset_roberta.py
@date: 10/07/2019

'''
import collections
import os
import pickle

import numpy as np
import json
from tqdm import tqdm

try:
    import regex as re
except Exception:
    import re

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "example_id", "tag", "logit"])

SPIECE_UNDERLINE = '▁'


class ChidExample(object):
    def __init__(self, example_id, tag, doc_tokens, options, answer_index=None):
        self.example_id = example_id
        self.tag = tag
        self.doc_tokens = doc_tokens
        self.options = options
        self.answer_index = answer_index

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "tag: %s" % (self.tag)
        s += ", context: %s" % (''.join(self.doc_tokens))
        s += ", options: [%s]" % (", ".join(self.options))
        if self.answer_index is not None:
            s += ", answer: %s" % self.options[self.answer_index]
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_id,
                 tag,
                 tokens,
                 input_ids,
                 input_masks,
                 segment_ids,
                 choice_masks,
                 label=None):
        self.unique_id = unique_id
        self.example_id = example_id
        self.tag = tag
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.choice_masks = choice_masks
        # The index of the correct answer among all candidate answers
        self.label = label


def read_chid_examples(input_data_file, input_label_file, is_training=True):
    '''
    The raw data is processed into the following form:
    part_passage traverse through every surroundings of the blank遍
    :param input_data:
    :param is_training:
    :return:
    '''

    if is_training:
        input_label = json.load(open(input_label_file))
    input_data = open(input_data_file)

    def _is_chinese_char(cp):
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

    def is_fuhao(c):
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’':
            return True
        return False

    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        is_blank = False
        for index, char in enumerate(text):
            cp = ord(char)
            if is_blank:
                output.append(char)
                if context[index - 12:index + 1].startswith("#idiom"):
                    is_blank = False
                    output.append(SPIECE_UNDERLINE)
            else:
                if text[index:index + 6] == "#idiom":
                    is_blank = True
                    if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                        output.append(SPIECE_UNDERLINE)
                    output.append(char)
                elif _is_chinese_char(cp) or is_fuhao(char):
                    if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                        output.append(SPIECE_UNDERLINE)
                    output.append(char)
                    output.append(SPIECE_UNDERLINE)
                else:
                    output.append(char)
        return "".join(output)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(
                c) == 0x202F or c == SPIECE_UNDERLINE:
            return True
        return False

    examples = []
    example_id = 0
    for data in tqdm(input_data):

        data = eval(data)
        options = data['candidates']

        for context in data['content']:

            context = context.replace("“", "\"").replace("”", "\"").replace("——", "--"). \
                replace("—", "-").replace("―", "-").replace("…", "...").replace("‘", "\'").replace("’", "\'")
            context = _tokenize_chinese_chars(context)

            paragraph_text = context.strip()
            doc_tokens = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False

            tags = [blank for blank in doc_tokens if '#idiom' in blank]

            if is_training:
                for tag_index, tag in enumerate(tags):
                    answer_index = input_label[tag]
                    example = ChidExample(
                        example_id=example_id,
                        tag=tag,
                        doc_tokens=doc_tokens,
                        options=options,
                        answer_index=answer_index)
                    examples.append(example)
            else:
                for tag_index, tag in enumerate(tags):
                    example = ChidExample(
                        example_id=example_id,
                        tag=tag,
                        doc_tokens=doc_tokens,
                        options=options)
                    examples.append(example)
        else:
            example_id += 1
    else:
        print('Original samples are ：{}'.format(example_id))

    print('The generated total examples are ：{}'.format(len(examples)))
    return examples


def add_tokens_for_around(tokens, pos, num_tokens):
    num_l = num_tokens // 2
    num_r = num_tokens - num_l

    if pos >= num_l and (len(tokens) - 1 - pos) >= num_r:
        tokens_l = tokens[pos - num_l:pos]
        tokens_r = tokens[pos + 1:pos + 1 + num_r]
    elif pos <= num_l:
        tokens_l = tokens[:pos]
        right_len = num_tokens - len(tokens_l)
        tokens_r = tokens[pos + 1:pos + 1 + right_len]
    elif (len(tokens) - 1 - pos) <= num_r:
        tokens_r = tokens[pos + 1:]
        left_len = num_tokens - len(tokens_r)
        tokens_l = tokens[pos - left_len:pos]
    else:
        raise ValueError('impossible')

    return tokens_l, tokens_r


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length=128,
                                 max_num_choices=10):
    '''
    Place all candidate answers at the beginning of the fragment
    '''

    def _loop(example, unique_id, label):
        '''
        :param example:
        :param unique_id:
        :return:
            input_ids = (C, seq_len)
            token_type_ids = (C, seq_len) = segment_id
            input_mask = (C, seq_len)
            labels = int
            choices_mask = (C)
        '''
        input_ids = []
        input_masks = []
        segment_ids = []
        choice_masks = [1] * len(example.options)

        tag = example.tag
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            if '#idiom' in token:
                sub_tokens = [str(token)]
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_doc_tokens.append(sub_token)

        pos = all_doc_tokens.index(tag)
        num_tokens = max_tokens_for_doc - 5
        tmp_l, tmp_r = add_tokens_for_around(all_doc_tokens, pos, num_tokens)
        num_l = len(tmp_l)
        num_r = len(tmp_r)

        tokens_l = []
        for token in tmp_l:
            if '#idiom' in token and token != tag:
                tokens_l.extend(['[MASK]'] * 4)
            else:
                tokens_l.append(token)
        tokens_l = tokens_l[-num_l:]
        del tmp_l

        tokens_r = []
        for token in tmp_r:
            if '#idiom' in token and token != tag:
                tokens_r.extend(['[MASK]'] * 4)
            else:
                tokens_r.append(token)
        tokens_r = tokens_r[:num_r]
        del tmp_r

        for i, elem in enumerate(example.options):
            option = tokenizer.tokenize(elem)
            tokens = ['[CLS]'] + option + ['[SEP]'] + tokens_l + [
                '[unused1]'
            ] + tokens_r + ['[SEP]']

            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            segment_id = [0] * len(input_id)

            while len(input_id) < max_seq_length:
                input_id.append(0)
                input_mask.append(0)
                segment_id.append(0)
            assert len(input_id) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_id) == max_seq_length

            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        if unique_id < 2:
            print("*** Example ***")
            print("unique_id: {}".format(unique_id))
            print("context_id: {}".format(tag))
            print("label: {}".format(label))
            print("tag_index: {}".format(pos))
            print("tokens: {}".format("".join(tokens)))
            print("choice_masks: {}".format(choice_masks))
        while len(input_ids) < max_num_choices:
            input_ids.append([0] * max_seq_length)
            input_masks.append([0] * max_seq_length)
            segment_ids.append([0] * max_seq_length)
            choice_masks.append(0)
        assert len(input_ids) == max_num_choices
        assert len(input_masks) == max_num_choices
        assert len(segment_ids) == max_num_choices
        assert len(choice_masks) == max_num_choices

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_id=example.example_id,
                tag=tag,
                tokens=tokens,
                input_ids=input_ids,
                input_masks=input_masks,
                segment_ids=segment_ids,
                choice_masks=choice_masks,
                label=label))

    # [CLS] choice [SEP] document [SEP]
    max_tokens_for_doc = max_seq_length - 3
    features = []
    unique_id = 0

    for (example_index, example) in enumerate(tqdm(examples)):

        label = example.answer_index
        if label != None:
            _loop(example, unique_id, label)
        else:
            _loop(example, unique_id, None)
        unique_id += 1

        if unique_id % 12000 == 0:
            print("unique_id: %s" % (unique_id))
    print("unique_id: %s" % (unique_id))
    return features


def logits_matrix_to_array(logits_matrix, index_2_idiom):
    """Compute the sequence with the highest global probability from a matrix"""
    logits_matrix = np.array(logits_matrix)
    logits_matrix = np.transpose(logits_matrix)
    tmp = []
    for i, row in enumerate(logits_matrix):
        for j, col in enumerate(row):
            tmp.append((i, j, col))
    else:
        choice = set(range(i + 1))
        blanks = set(range(j + 1))
    tmp = sorted(tmp, key=lambda x: x[2], reverse=True)
    results = []
    for i, j, v in tmp:
        if (j in blanks) and (i in choice):
            results.append((i, j))
            blanks.remove(j)
            choice.remove(i)
    results = sorted(results, key=lambda x: x[1], reverse=False)
    results = [[index_2_idiom[j], i] for i, j in results]
    return results


def logits_matrix_max_array(logits_matrix, index_2_idiom):
    logits_matrix = np.array(logits_matrix)
    arg_max = logits_matrix.argmax(axis=1)
    results = [[index_2_idiom[i], idx] for i, idx in enumerate(arg_max)]
    return results


def get_final_predictions(all_results, tmp_predict_file, g=True):
    if not os.path.exists(tmp_predict_file):
        pickle.dump(all_results, open(tmp_predict_file, 'wb'))

    raw_results = {}
    for i, elem in enumerate(all_results):
        example_id = elem.example_id
        if example_id not in raw_results:
            raw_results[example_id] = [(elem.tag, elem.logit)]
        else:
            raw_results[example_id].append((elem.tag, elem.logit))

    results = []
    for example_id, elem in raw_results.items():
        index_2_idiom = {index: tag for index, (tag, logit) in enumerate(elem)}
        logits = [logit for _, logit in elem]
        if g:
            results.extend(logits_matrix_to_array(logits, index_2_idiom))
        else:
            results.extend(logits_matrix_max_array(logits, index_2_idiom))
    return results


def write_predictions(results, output_prediction_file):

    results_dict = {}
    for result in results:
        results_dict[result[0]] = result[1]
    with open(output_prediction_file, 'w') as w:
        json.dump(results_dict, w, indent=2)

    print("Writing predictions to: {}".format(output_prediction_file))


def generate_input(data_file,
                   label_file,
                   example_file,
                   feature_file,
                   tokenizer,
                   max_seq_length,
                   max_num_choices,
                   is_training=True):
    if os.path.exists(feature_file):
        features = pickle.load(open(feature_file, 'rb'))
    elif os.path.exists(example_file):
        examples = pickle.load(open(example_file, 'rb'))
        features = convert_examples_to_features(examples, tokenizer,
                                                max_seq_length, max_num_choices)
        pickle.dump(features, open(feature_file, 'wb'))
    else:
        examples = read_chid_examples(
            data_file, label_file, is_training=is_training)
        pickle.dump(examples, open(example_file, 'wb'))
        features = convert_examples_to_features(examples, tokenizer,
                                                max_seq_length, max_num_choices)
        pickle.dump(features, open(feature_file, 'wb'))

    return features


def evaluate(ans_f, pre_f):
    ans = json.load(open(ans_f))
    pre = json.load(open(pre_f))

    total_num = 0
    acc_num = 0
    for id_ in ans:
        if id_ not in pre:
            raise FileNotFoundError
        total_num += 1
        if ans[id_] == pre[id_]:
            acc_num += 1

    acc = acc_num / total_num
    acc *= 100
    return acc
