import argparse
import yaml
from attrdict import AttrDict
from pprint import pprint
import requests
import json
import os, sys
import time
from paddlenlp.datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/transformer.big.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument("--batch-size", type=int, help="Batch size. ")
    parser.add_argument(
        "--profile", action="store_true", help="Whether to profile. ")
    args = parser.parse_args()
    return args


def do_client(args):
    dataset = load_dataset('wmt14ende', splits=('test'))

    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:9292/transformer/prediction"

    batch = []
    samples = 0
    f = open(args.output_file, "w")
    # f.write(sequence + "\n")

    for sequence in dataset:
        samples += 1
        if len(batch) < args.infer_batch_size:
            batch.append(sequence[args.src_lang])
            continue
        data = {"feed": [{"src_word": batch}], "fetch": ["finished_sequence"]}
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        for seq in r.json()["result"]["finished_sequence"]:
            print(seq)
            f.write(seq[0] + "\n")
        batch = []
    f.close()


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_client(args)
