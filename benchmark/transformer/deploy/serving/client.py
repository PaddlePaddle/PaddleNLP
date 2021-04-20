import argparse
import yaml
from attrdict import AttrDict
from pprint import pprint
import requests
import json
import os, sys
import time
from paddlenlp.datasets import load_dataset
from utils.recorder import Recorder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../configs/transformer.big.yaml",
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
    sample = 0
    f = open(args.output_file, "w")
    if args.profile:
        recorder = Recorder(args.infer_batch_size, args.model_name)
        recorder.tic()

    for sequence in dataset:
        sample += 1
        batch.append(sequence[args.src_lang])
        if len(batch) < args.infer_batch_size and sample != len(dataset):
            continue
        data = {"feed": [{"src_word": batch}], "fetch": ["finished_sequence"]}
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        print(r)

        if args.profile:
            recorder.toc(samples=len(batch))
        batch = []
        for seq in r.json()["result"]["finished_sequence"]:
            f.write(seq[0] + "\n")
        if args.profile:
            recorder.tic()
    if args.profile:
        recorder.report(model_name=args.model_name)
    f.close()


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    if ARGS.batch_size is not None:
        args.infer_batch_size = ARGS.batch_size
    args.profile = ARGS.profile
    args.model_name = "transformer_base" if "base" in ARGS.config else "transformer_big"

    do_client(args)
