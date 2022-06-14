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
import json
import time
import re


class Timer(object):
    """Stat Cost Time"""

    def __init__(self, msg=""):
        super(Timer, self).__init__()

        self._msg = msg
        self._start = time.time()
        self._last = self._start

    def reset(self, only_last=False, msg=None):
        """reset all setting
        """
        if msg is not None:
            self._msg = msg
        curr_time = time.time()
        self._last = curr_time
        if not only_last:
            self._start = curr_time

    def check(self):
        """check cost time from start
        """
        end = time.time()
        cost = end - self._start
        return cost

    def interval(self):
        """check cost time from lst
        """
        end = time.time()
        cost = end - self._last
        self._last = end
        return cost

    def ending(self):
        """ending checking and log
        """
        cost = '%.2f' % time.time() - self._start
        if self._msg == "":
            log_msg = "cost time: %s" % (cost)
        elif '{}' in self._msg:
            log_msg = self._msg.format(cost)
        else:
            log_msg = self._msg + cost

        logging.info(log_msg)


def list_increment(lst: list, base: int):
    """increment each element in list
    """
    for i in range(len(lst)):
        lst[i] += base
    return lst


def count_file_lines(filename):
    cnt = 0
    with open(filename) as ifs:
        for _ in ifs:
            cnt += 1
    return cnt


def print_tensors(tag='*', **kwrags):
    """print tensors for debuging
    """
    print(tag * 50)
    for key, value in kwrags.items():
        print(key, ':', value)


if __name__ == "__main__":
    """run some simple test cases"""
    import json
    from boomup import data_struct
    question = '三峡碧江需要大于2的招聘数量'
    table_json = {
        "rows": [[
            4.0, "污水运行工", "三峡碧江公司", "渝北", 2.0, "大专及以上", "给排水/环境工程/机电及相关专业",
            "sxswrlzyb@163.com"
        ],
                 [
                     5.0, "污水运行工", "三峡垫江公司", "垫江", 1.0, "大专及以上",
                     "给排水/环境工程/机电及相关专业", "sxswrlzyb@163.com"
                 ]],
        "name":
        "Table_a7b5108c3b0611e98ad7f40f24344a08",
        "title":
        "",
        "header":
        ["岗位序号", "招聘岗位", "用人单位", "工作地点", "招聘数量", "学历要求", "专业及资格要求", "简历投递邮箱"],
        "common":
        "",
        "id":
        "a7b510",
        "types":
        ["real", "text", "text", "text", "real", "text", "text", "text"]
    }
    table_json['header'] = data_struct.Header(table_json['header'],
                                              table_json['types'])
    table = data_struct.Table(**table_json)
