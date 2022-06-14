#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
import re
from collections import defaultdict

from statistics import mean
import cn2an
from LAC import LAC

g_max_candi_value = 5
g_date_patt = re.compile(r'(([0-9]{2})[0-9]{2}-)?(0?[1-9]|1[012])-[0123][0-9]')
g_date_patt2 = re.compile(
    r'(([0-9]{2})[0-9]{2}年)?[0-9]{1,2}月[0-9]{2}[号日]|([0-9]{2})[0-9]{2}年[0-9]{1,2}月'
)

g_lac_seg = LAC(mode='seg')
g_lac_lac = LAC(mode='lac')

wordseg = lambda sentence: g_lac_seg.run(sentence)
lac = lambda sentence: g_lac_lac.run(sentence)

## LAC Tags
# 标签 含义      标签 含义      标签 含义       标签 含义
# n    普通名词  f    方位名词  s    处所名词   nw   作品名
# nz   其他专名  v    普通动词  vd   动副词     vn   名动词
# a    形容词    ad   副形词    an   名形词     d    副词
# m    数量词    q    量词      r    代词       p    介词
# c    连词      u    助词      xc   其他虚词   w    标点符号
# PER  人名      LOC  地名      ORG  机构名     TIME 时间
g_ner_tag_mapping = {
    'LOC': 'LOC',
    'TIME': 'TIME',
    'PER': 'PER',
    'm': 'NUM',
}
EMPTY_TAG = 'o'


def ner(sentence):
    """wordseg and ner

    Args:
        sentence (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    results = lac(sentence)
    words = results[0]
    tags_tmp = results[1]
    tags = []
    for tag in tags_tmp:
        tags.append(g_ner_tag_mapping.get(tag, EMPTY_TAG))
    return (words, tags)


def ngrams(tok_list, n):
    """generate n-grams from tok_list

    Args:
        tok_list (TYPE): NULL
        n (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    for pos in range(len(tok_list) - n + 1):
        yield tok_list[pos:pos + n]


def remove_brackets(s):
    """Remove brackets [] () from text
    """
    return re.sub(r'[\(\（].*[\)\）]', '', s)


def is_float(value):
    """is float"""
    try:
        float(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def cn_to_an(string):
    """cn to an"""
    try:
        return str(cn2an.cn2an(string, 'normal'))
    except ValueError:
        return string


def an_to_cn(string):
    """an to cn"""
    try:
        return str(cn2an.an2cn(string))
    except ValueError:
        return string


def str_to_num(string):
    """str to num"""
    try:
        float_val = float(cn_to_an(string))
        if int(float_val) == float_val:
            return str(int(float_val))
        else:
            return str(float_val)
    except ValueError:
        return None


def str_to_year(string):
    """str to year"""
    year = string.replace('年', '')
    year = cn_to_an(year)
    if is_float(year) and float(year) < 1900:
        year = int(year) + 2000
        return str(year)
    else:
        return None


class CandidateValueExtractor:
    """
    params:
    """
    CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两１２３４５６７８９０'
    CN_UNIT = '十拾百佰千仟万萬亿億兆点．'

    @classmethod
    def norm_unit(cls, rows, col_id, values):
        """norm unit"""
        l = []
        for row in rows:
            if isinstance(row[col_id], str) or row[col_id] is None:
                return None
            l.append(len(str(int(row[col_id]))))
        mean_len = round(mean(l) + 0.5)

        new_values = set()
        for value in values:
            flag = False
            if value.isdigit():
                str_value = str(value)
                diff = len(str_value) - mean_len
                if diff > 2:
                    tail_str = str_value[-1 * diff:]
                    if tail_str.count('0') == len(tail_str):
                        new_values.add(str_value[:mean_len])
                        new_values.add(value)
                        flag = True
            if not flag:
                new_values.add(value)
        return list(new_values)

    @classmethod
    def search_values(cls, question, table):
        """search candidate cells from table, that will be used as sql values

        Args:
            question_words (list): NULL
            question_tags (list): NULL
            table (Table): NULL

        Returns: TODO

        Raises: NULL
        """
        # 提取年份和数字
        value_in_question = cls.extract_values_from_text(question)
        all_candidate = []
        for col_id in range(len(table.header)):
            header = table.header[col_id]
            # 提取col出现在quesiton中的cell
            # TODO 这里存在一个问题，一个text类型cell必须完全在question中出现才会被当做候选cell
            value_in_column = cls.extract_values_from_column(
                question, table, col_id, header.type)
            if header.type == 'text':
                candi_values = value_in_column
            elif header.type == 'real':
                norm_unit_res = cls.norm_unit(table.rows, col_id,
                                              value_in_question)
                if norm_unit_res is not None:
                    value_in_question = norm_unit_res
                candi_values = value_in_question
                if len(candi_values) >= g_max_candi_value:
                    candi_values = candi_values[:g_max_candi_value]
                else:
                    st_candi_values = set(candi_values)
                    for v in value_in_column:
                        if v in st_candi_values:
                            continue
                        st_candi_values.add(v)
                        if len(st_candi_values) >= g_max_candi_value:
                            break
                    candi_values = list(st_candi_values)
            all_candidate.append(candi_values)
        return all_candidate

    # 19年 or 一九年 will be replaced to 2019年
    @classmethod
    def extract_year_from_text(cls, text):
        """extract year from text"""
        values = []
        # FIXME trick: yrs is from 2000
        num_year_texts = re.findall(r'[0-9][0-9]年', text)
        values += ['20{}'.format(text[:-1]) for text in num_year_texts]
        cn_year_texts = re.findall(r'[{}][{}]年'.format(cls.CN_NUM, cls.CN_NUM),
                                   text)
        cn_year_values = [str_to_year(text) for text in cn_year_texts]
        values += [value for value in cn_year_values if value is not None]
        return values

    @classmethod
    def extract_date_from_text(cls, text):
        """

        Args:
            text (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        date_values = []
        unmatched_spans = []
        base = 0
        while base < len(text):
            res = re.search(g_date_patt, text[base:])
            if res is None:
                unmatched_spans.append(text[base:])
            else:
                start, end = res.span()
                unmatched_spans.append(text[base:start])
                unmatched_spans.append(text[end:])
                base = end
                date_values.append(text[start:end])
        return date_values, list(
            filter(lambda x: x.strip() != '', unmatched_spans))

    @classmethod
    def extract_num_from_text(cls, text):
        """extract num from text"""
        values = []
        # 1. all digital number
        num_values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)
        values += num_values
        # 2. include chinese word
        cn_num_unit = cls.CN_NUM + cls.CN_UNIT
        cn_num_texts = re.findall(
            r'[{}]*\.?[{}]+'.format(cn_num_unit, cn_num_unit), text)

        cn_num_values = [str_to_num(text) for text in cn_num_texts]
        values += [value for value in cn_num_values if value is not None]
        # 3. both number and chinese word
        cn_num_mix = re.findall(r'[0-9]*\.?[{}]+'.format(cls.CN_UNIT), text)
        for word in cn_num_mix:
            num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', word)
            for n in num:
                word = word.replace(n, an_to_cn(n))
            str_num = str_to_num(word)
            if str_num is not None:
                values.append(str_num)
        return values

    @classmethod
    def extract_values_from_text(cls, text):
        """extract values from text"""
        values = []
        values += cls.extract_year_from_text(text)
        values_tmp, unmatched_spans = cls.extract_date_from_text(text)
        values.extend(values_tmp)
        for span in unmatched_spans:
            values += cls.extract_num_from_text(span)
        return list(set(values))

    @classmethod
    def extract_values_from_column(cls, question, table, col_id, col_type):
        """extract values from column"""
        if col_type == "text":
            base_threshold = 0
        else:
            base_threshold = 2

        value_score = table.search(question, col_id)
        value_score_filter = list(
            filter(lambda x: x[1] > base_threshold, value_score))
        if len(value_score_filter) == 0:
            return []

        value_score_filter.sort(key=lambda x: x[1], reverse=True)
        ##if col_type == 'text' \
        ##        and len(value_score_filter) > g_max_candi_value \
        ##        and value_score_filter[g_max_candi_value][1] == value_score_filter[0][1]:
        ##    value_score_filter_tmp = value_score_filter[:50]
        ##    tmp_score = value_score_filter[g_max_candi_value][1]
        ##    select_col_values = [x[0] for x in value_score_filter_tmp if x[1] >= tmp_score]
        ##else:
        ##    select_col_values = [x[0] for x in value_score_filter[:g_max_candi_value]]
        select_col_values = [
            x[0] for x in value_score_filter[:g_max_candi_value]
        ]

        return select_col_values


def re_search(patt, text):
    """

    Args:
        patt (TYPE): NULL
        text (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    lst_result = []
    pos = 0
    while True:
        match = re.search(patt, text[pos:])
        if match is None:
            break
        lst_result.append((match.start() + pos, match.end() + pos))
        pos = pos + match.end() + 1

    return lst_result


CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两１２３４５６７８９０'
CN_UNIT = '十拾百佰千仟万萬亿億兆点．'
CN_NUM_UNIT = CN_NUM + CN_UNIT
PATT_NUM = re.compile(r'[-+]?[0-9]*\.?[0-9]+')
PATT_CN_NUM = re.compile(r'[{}]*\.?[{}]+'.format(CN_NUM_UNIT, CN_NUM_UNIT))
PATT_MIX_NUM = re.compile(r'[0-9]*\.?[{}]+'.format(CN_UNIT))


def _extract_num_span(text):
    """extract number and mark their spans

    Args:
        text (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    dct_start2end = defaultdict(set)
    # digital number
    spans = re_search(PATT_NUM, text)
    for start, end in spans:
        dct_start2end[start].add((end, text[start:end]))

    # chinese number
    spans = re_search(PATT_CN_NUM, text)
    for start, end in spans:
        num = str_to_num(text[start:end])
        if num is None:
            continue
        dct_start2end[start].add((end, num))

    # number, chinese
    spans = re_search(PATT_MIX_NUM, text)
    for start, end in spans:
        orig_num = text[start:end]
        for ar_num in re.findall(PATT_NUM, orig_num):
            orig_num = orig_num.replace(ar_num, an_to_cn(ar_num))
        num = str_to_num(orig_num)
        if num is not None:
            dct_start2end[start].add((end, num))

    lst_result = []
    for start, st_end_and_num in sorted(dct_start2end.items()):
        lst_end, lst_num = list(zip(*st_end_and_num))
        end = max(lst_end)
        if len(lst_result) == 0 or start > lst_result[-1][0][1]:
            lst_result.append([(start, end), lst_num])
            continue
        last_start, last_end = lst_result[-1][0]
        if end - start > last_end - last_start:
            lst_result.pop(-1)
            lst_result.append([(start, end), lst_num])
        else:
            pass

    return lst_result


def wordseg_and_extract_num(text):
    lst_span_and_nums = _extract_num_span(text)
    lst_words = []
    pos = 0
    lst_nums = []
    lst_nums_index = []
    for span, nums in lst_span_and_nums:
        start, end = span
        lst_words.extend(wordseg(text[pos:start]))
        lst_nums.extend(nums)
        lst_nums_index.extend([len(lst_words)] * len(nums))
        lst_words.append(text[start:end])
        pos = end

    if pos < len(text):
        lst_words.extend(wordseg(text[pos:]))

    return lst_words, lst_nums, lst_nums_index


if __name__ == "__main__":
    """run some simple test cases"""
    lst_token = ['hello', ',', 'I', 'am', 'Li', 'Lei', '.']
    print(list(ngrams(lst_token, 2)))
    print(list(ngrams(lst_token, 4)))

    text = '123年后，你好一百万不多，2百万不少'
    print(wordseg_and_extract_num(text))
