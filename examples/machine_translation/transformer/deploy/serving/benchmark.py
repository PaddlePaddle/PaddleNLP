import sys
import os
import yaml
import requests
import time
import json
import ast
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args, show_latency


def parse_benchmark(filein, fileout):
    with open(filein, "r") as fin:
        res = yaml.load(fin)
        del_list = []
        for key in res["DAG"].keys():
            if "call" in key:
                del_list.append(key)
        for key in del_list:
            del res["DAG"][key]
    with open(fileout, "w") as fout:
        yaml.dump(res, fout, default_flow_style=False)


if __name__ == "__main__":
    filein = sys.argv[1]
    fileout = sys.argv[2]
    parse_benchmark(filein, fileout)
