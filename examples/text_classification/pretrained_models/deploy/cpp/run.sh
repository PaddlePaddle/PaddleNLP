#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. run
./build/seq_cls_infer --model_file ./export/inference.pdmodel --params_file ./export/inference.pdiparams --vocab_file ./vocab.txt
