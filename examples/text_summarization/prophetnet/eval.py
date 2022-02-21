import argparse
import os
import re
import sys
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="choose from all, or 1 of 8 dataset like cnndm, gigaword etc.")
parser.add_argument("--generated", type=str, help="generated output file.")

args = parser.parse_args()

data_root_path = 'data'

support_dataset = ['cnndm', 'gigaword']
files2rouge_template = '.*ROUGE-1 Average_F: (?P<rouge1_f>\d+(\.\d*)?|\.\d+).*ROUGE-2 Average_F: (?P<rouge2_f>\d+(\.\d*)?|\.\d+).*ROUGE-L Average_F: (?P<rougeL_f>\d+(\.\d*)?|\.\d+).*'
# gigaword_template='.*ROUGE-1: (?P<rouge1_f>\d+(\.\d*)?|\.\d+).*ROUGE-2: (?P<rouge2_f>\d+(\.\d*)?|\.\d+).*ROUGE-L: (?P<rougeL_f>\d+(\.\d*)?|\.\d+).*'
qg_template = '.*Bleu_4: (?P<bleu4>\d+(\.\d*)?|\.\d+).*METEOR: (?P<meteor>\d+(\.\d*)?|\.\d+).*ROUGE_L: (?P<rougeL>\d+(\.\d*)?|\.\d+).*'
personachat_template = '.*?(?P<d1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?).*?(?P<d2>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?).*Bleu_1: (?P<bleu1>\d+(\.\d*)?|\.\d+).*Bleu_2: (?P<bleu2>\d+(\.\d*)?|\.\d+).*'


def scale_up(d):
    return {k: float(d[k]) * 100 for k in d.keys()}


def eval_one_dataset():
    golden_file = f"{data_root_path}/{args.dataset}_data/test.tgt"

    eval_template = {
        'cnndm':
        f"python ./evaluate/cnndm/postprocess_cnn_dm.py --generated {generated_file} --golden {golden_file}",
        'gigaword':
        f"python ./evaluate/gigaword/eval.py --perl --pred {generated_file} --gold {golden_file}",
    }

    cmd = eval_template[args.dataset]
    try:
        output = os.popen(cmd).read()
        if args.dataset in ['cnndm', 'gigaword']:
            d = re.search(files2rouge_template,
                          output.replace("\n", " ")).groupdict()
            d = scale_up(d)
            print(
                f"{args.dataset}\trouge1/rouge2/rougeL\t{d['rouge1_f']:.2f}/{d['rouge2_f']:.2f}/{d['rougeL_f']:.2f}"
            )
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(f"{args.dataset} evaluate failed!")


if args.dataset != 'all':
    generated_file = args.generated
    eval_one_dataset()
else:
    output_root_path = args.generated
    onlyfolders = [
        f for f in listdir(output_root_path)
        if not isfile(join(args.generated, f))
    ]
    for dataset in support_dataset:
        for folder in onlyfolders:
            if folder.startswith(dataset):
                for hypo_file in listdir(args.generated + '/' + folder):
                    if 'hypo' in hypo_file or 'score' in hypo_file:
                        generated_file = args.generated + '/' + folder + '/' + hypo_file
                        print(f"{dataset}\tpredict_file:{generated_file}")
                        args.dataset = dataset
                        args.gnerated = generated_file
                        eval_one_dataset()
