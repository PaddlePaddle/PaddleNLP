#!/usr/bin/python3

#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import os
import subprocess

from util import get_model_path

realwd = os.path.dirname(os.path.realpath(__file__))
paddlenlp_path = os.path.realpath(f"{realwd}/../../")

data_path = os.environ.get("DATA_DIR", "/data")

os.environ.setdefault("GAUDI2_CI", "1")

cmd_args = {}
parser = argparse.ArgumentParser(
    description="help scriopt for pdpd e2e run test on intel hpu",
    add_help=True,
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--context",
    choices=["pr", "bat", "sanity", "full"],
    help="which test suites to be used: pr(PR testing), bat (BAT testing), sanity (Smoke testing), full (full testing); default: bat",
    default="bat",
)
parser.add_argument("--data", type=str, help="data folder which should include huggingface folder", default=data_path)
parser.add_argument(
    "--filter",
    choices=["stable", "unstable", "all"],
    help="filter test case list: stable/unstable/all",
    default="all",
)
parser.add_argument("--device", type=str, help="device name", default="intel_hpu")
parser.add_argument("--mode", type=str, help="it should be one of [dynamic, static]", default="dynamic")
parser.add_argument("--junit", type=str, help="junit result file")
parser.add_argument("--platform", type=str, help="platform name")

cmd_args.update(vars(parser.parse_args()))

if cmd_args["junit"]:
    libpath = os.path.dirname(os.path.dirname(realwd))
    if os.path.exists(f"{libpath}/junitxml.py"):
        import sys

        sys.path.append(libpath)
    from junitxml import jTestCase, jTestSuite

cmd_args["platform"] = "" if cmd_args["platform"] is None else cmd_args["platform"].lower()

script_path = os.path.dirname(os.path.realpath(__file__))

if os.path.exists(cmd_args["data"]) is False:
    print("data path not exist, please check for parameter: data / environment DATA_DIR")
    exit(2)

data_path = f"{cmd_args['data']}"
os.environ.setdefault("DATA_HOME", data_path)
if os.path.exists(data_path) is False:
    print(f"Couldn't find mode data path, please confirm this folder under the {cmd_args['data']} folder")
    exit(2)


def case_command(command, test_case=None, test_suite=None):
    output = ""
    output += f"Command: {command}\n"
    proc = subprocess.Popen(command, bufsize=0, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        line = proc.stdout.readline().decode()
        while len(line) > 0:
            print(f"{line[:-1]}")  # line include '\n' at last
            output += line
            line = proc.stdout.readline().decode()
    except Exception as e:
        if test_case:
            test_case.setFail("Command abnormally")
        print(e)
    finally:
        proc.communicate()
    if proc.returncode != 0 and test_case:
        test_case.setFail(f"Command return code:{proc.returncode}")
    if test_case:
        test_case.AddOutput(output)
    if test_suite:
        ts.addCase(test_case)

    return proc.returncode, output


def run_e2e_test_case(
    model,
    model_case_name,
    src_length,
    max_length,
    total_max_length,
    batch_size,
    decode_strategy,
    output_file,
    skip_lst=None,
    test_suite=None,
    cmd_env="",
):
    # mode option: dynamic or static
    mode_opt = ""
    mode_param = ""
    if cmd_args["mode"] and cmd_args["mode"] in ["dynamic", "static"]:
        mode_opt = f"--mode {cmd_args['mode']}"
        mode_param = cmd_args["mode"]
    else:
        mode_param = "dynamic"

    # device option
    device_opt = "intel_hpu"
    if cmd_args["device"]:
        device_opt = cmd_args["device"]

    # gaudi2d platform only support bfloat16
    float_opt = "--dtype bfloat16" if cmd_args["platform"] == "gaudi2d" else ""

    testcase = None
    testcase_name = (
        f"{model_case_name}-{src_length}-{max_length}-{total_max_length}-{batch_size}-{decode_strategy}-{mode_param}"
    )
    if test_suite:
        testcase = jTestCase(testcase_name)
    else:
        testcase = None

    cmd_line = f"python {paddlenlp_path}/predict/predictor.py --model_name_or_path {model} --inference_model --device {device_opt} {mode_opt} {float_opt} "
    cmd_line += f"--src_length {src_length} --max_length {max_length} --total_max_length {total_max_length} --batch_size {batch_size} --decode_strategy {decode_strategy} --output_file result/{testcase_name}.json"
    _env = os.environ.copy()
    for opt in cmd_env.split():
        _env.setdefault(opt.split("=")[0], opt.split("=")[1])
    print(f"RUN shell CMD: {cmd_env} {cmd_line}")
    ret, output = case_command(cmd_line, testcase, test_suite)

    return ret, output


case_dict_lst = {}

skip_lst = []

from config.llm import skip_case_lst, test_case_lst

case_dict_lst = test_case_lst[cmd_args["context"]]
skip_lst = skip_case_lst.get(cmd_args["context"], {}).get(cmd_args["filter"], [])

ts = None
if cmd_args["junit"]:
    ts = jTestSuite("E2E Test")
    ts.setPlatform(cmd_args["platform"])

total_case_num = 0
pass_case_num = 0
fail_case_num = 0

for model_name, case_dict in case_dict_lst.items():
    model_case_name = model_name.split("/")[-1]
    model_path_or_name = get_model_path(model_name)
    variants = case_dict.get("variants", dict()).get("inference", [])
    mode_param = "dynamic"
    if cmd_args["mode"] and cmd_args["mode"] in ["dynamic", "static"]:
        mode_param = cmd_args["mode"]

    for variant_dict in variants:
        case_tag = f"{model_name} src:{variant_dict['src_length']}-max_length:{variant_dict['max_length']}-total_max_length:{variant_dict['total_max_length']}-bs:{variant_dict['batch_size']}-decode_strategy:{variant_dict['decode_strategy']}-mode:{mode_param}"
        ret_test, _ = run_e2e_test_case(
            model_path_or_name,
            model_case_name,
            variant_dict["src_length"],
            variant_dict["max_length"],
            variant_dict["total_max_length"],
            variant_dict["batch_size"],
            variant_dict["decode_strategy"],
            case_dict["output_file"],
            skip_lst,
            ts,
        )
        if "skip" == _:
            pass
        else:
            total_case_num = total_case_num + 1
            if ret_test == 0:
                pass_case_num = pass_case_num + 1
                print(f"\033[0;32mtest case {total_case_num} : {case_tag} pass \033[0m")
            else:
                fail_case_num = fail_case_num + 1
                print(f"\033[0;31mtest case {total_case_num} : {case_tag} fail \033[0m")

if cmd_args["junit"]:
    with open(cmd_args["junit"], "w+") as f:
        ts.toString()
        f.write(ts.toString())

print("...............................Summary.......................................")
print(f"\033[0;37mE2E total {total_case_num} test case running \033[0m")
print(f"\033[0;32mE2E total {pass_case_num} test case pass \033[0m")
if fail_case_num != 0:
    print(f"\033[0;31mE2E total {fail_case_num} test case fail \033[0m")

exit(fail_case_num)
