#!/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
import re
import subprocess
from enum import Enum

from lxml import etree as et


class jResult(Enum):
    PASS = 0
    FAIL = 1
    SKIP = 2
    ERROR = 3
    DISABLE = 4


def remove_invalidel_xml_characters(src, output, start=10):
    # "invalid XML character" will return ""
    numerical = int(src, start)
    if (
        numerical in (0xB, 0xC, 0xFFFE, 0xFFFF)
        or 0x0 <= numerical <= 0x8
        or 0xE <= numerical <= 0x1F
        or 0xD800 <= numerical <= 0xDFFF
    ):
        return ""
    return output


class jXMLBase:
    def __init__(self, tag: str):
        self._attr_dict = dict()
        self._tag = tag
        self._text = ""

    def setName(self, name):
        self._attr_dict["name"] = name

    def covertET(self) -> et:
        elem = et.Element(self._tag)
        for k, v in self._attr_dict.items():
            elem.set(k, str(v))
        if len(self._text) > 0:
            # to deal all non-ascii characters for XML
            self._text = re.sub(
                r"&#(\d+);?", lambda ch: remove_invalidel_xml_characters(ch.group(1), ch.group(0)), self._text
            )
            self._text = re.sub(
                r"&#[xX]([0-9a-fA-F]+);?",
                lambda ch: remove_invalidel_xml_characters(ch.group(1), ch.group(0), start=16),
                self._text,
            )
            self._text = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]").sub("", self._text)
            elem.text = self._text
        return elem

    def toString(self, pretty_print=False):
        return et.tostring(self.covertET(), encoding="utf-8", pretty_print=pretty_print).decode()


class jTestCase(jXMLBase):
    result = jResult.PASS
    _child_elem = None
    _output = None

    def __init__(self, name: str):
        super().__init__("testcase")
        self.setName(name)
        self._child_elem = list()

    def setClass(self, classname: str):
        self._attr_dict["classname"] = classname

    def setTime(self, time):
        self._attr_dict["time"] = time

    def AddOutput(self, output: str):
        if self._output is None:
            self._output = jXMLBase("system-out")
            self._child_elem.append(self._output)
        if len(output) >= 1:
            self._output._text += output

    def setFail(self, message: str):
        self.result = jResult.FAIL
        el = jXMLBase("failure")
        el._attr_dict["message"] = message
        el._attr_dict["type"] = "failure"
        self._child_elem.append(el)

    def setSkip(self, message: str):
        self.result = jResult.SKIP
        self._attr_dict["skipped"] = 1
        self.AddOutput("")
        el = jXMLBase("skipped")
        el._attr_dict["type"] = message
        self._child_elem.append(el)

    def covertET(self) -> et:
        elem = super().covertET()
        for jxml in self._child_elem:
            elem.append(jxml.covertET())
        return elem


class jTestSuite(jXMLBase):
    _case_lst = None

    def __init__(self, name: str):
        super().__init__("testsuite")
        self.setName(name)
        self._case_lst = list()
        for i in ["tested", "passed", "errors", "failed", "skipped", "disabled"]:
            self._attr_dict[i] = 0
        self._attr_dict["kernel"] = subprocess.getoutput("uname -r")
        self._attr_dict["os"] = "wsl" if "microsoft" in platform.platform().lower() else "linux"

    def print_attr_dict(self):
        for k, v in self._attr_dict.items():
            print(f"{k} : {v}")

    def setPlatform(self, platform):
        self._attr_dict["platform"] = platform

    def addCase(self, case: jTestCase):
        if case.result == jResult.FAIL:
            self._attr_dict["failed"] += 1
        elif case.result == jResult.ERROR:
            self._attr_dict["errors"] += 1
        elif case.result == jResult.SKIP:
            self._attr_dict["skipped"] += 1
        elif case.result == jResult.DISABLE:
            self._attr_dict["disabled"] += 1
        else:
            self._attr_dict["passed"] += 1
        self._case_lst.append(case)

    def covertET(self) -> et:
        self._attr_dict["tested"] = len(self._case_lst)
        elem = super().covertET()
        for jxml in self._case_lst:
            elem.append(jxml.covertET())
        return elem
