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

import argparse
import re
import sys


def parameter_parser():
    parser = argparse.ArgumentParser(description="Support Args:")
    parser.add_argument("-v", "--valid-expr", type=str, default="*", help="when not match, the line will discard.")
    parser.add_argument(
        "-e", "--extract-expr", type=str, default="^{%s}$,", help="the extract expr for the loss: loss {%f}"
    )
    parser.add_argument("-r", "--reduction-expr", type=str, default="print", help="print | sum | mean")
    parser.add_argument("-n", "--discard", type=int, default=0, help="while reduction, discard [0:n] and [-n:]")
    parser.add_argument("-d", "--debug", type=bool, default=False, help="debug")
    return parser.parse_args()


args = parameter_parser()


def log(*inp, **kargs):
    if args.debug:
        print(*inp, **kargs)


def is_valid(line, valid_expr):
    if valid_expr == "*":
        return True
    if valid_expr in line:
        return True
    return False


def extract(line, extract_expr):
    """
    return tuple, the output will be
    """
    log("Extract_expression is : ", extract_expr)
    x = re.findall("\{%(.)\}", extract_expr)
    assert len(x) == 1, "Must exist a {%d} | {%f} | {%s} "
    t = x[0]
    type_converter = {
        "f": float,
        "i": int,
        "s": str,
    }
    type_extracter = {
        "f": r"(-?\\d+\\.\\d+)",
        "i": r"(-?\\d+)",
        "s": r"(.*?)",
    }
    log(type_extracter[t])
    pattern = re.sub("\{%(.)\}", type_extracter[t], extract_expr, 1)
    log("Created Pattern is: ", pattern)
    x = re.findall(pattern, line)
    if len(x) == 0:
        return None
    assert len(x) == 1, f"Multi Match for `{extract_expr}` in line: \n{line}"
    log("Find in line: ", x[0].strip())
    return type_converter[t](x[0].strip())


def action(tuple_list, action):
    # discard the warm up
    if args.discard > 0:
        tuple_list = tuple_list[args.discard :]
        tuple_list = tuple_list[: -args.discard]
    # do action for each item
    if action == "sum":
        print(sum(tuple_list))
    if action == "mean":
        if len(tuple_list) == 0:
            print("null")
        else:
            print(sum(tuple_list) / len(tuple_list))
    if action == "print":
        for item in tuple_list:
            print(item)


def main():
    tuple_list = []
    for line in sys.stdin:
        line = line.strip()
        if is_valid(line, args.valid_expr):
            ret = extract(line, args.extract_expr)
            if ret:
                tuple_list.append(ret)
    action(tuple_list, args.reduction_expr)


if __name__ == "__main__":
    main()
