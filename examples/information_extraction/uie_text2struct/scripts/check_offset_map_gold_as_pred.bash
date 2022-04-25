#!/usr/bin/env bash
# -*- coding:utf-8 -*-

# Check Offset Mapping Performance
# 用于验证不同 SEL2Record 回标策略的准确值
# bash scripts/check_offset_map_gold_as_pred.bash data/text2spotasocname/absa/14lap config/offset_map/closest_offset_en.yaml spotasocname

folder_name=$1
config_name=$2
parser_format=$3

cat ${folder_name}/val.json | python -c "import json, sys
for line in sys.stdin:
    print(json.loads(line.strip())['record'])
" > ${folder_name}/eval_preds_seq2seq.txt

python scripts/sel2record.py \
    -c ${config_name} \
    -g ${folder_name} \
    -p ${folder_name} \
    -d ${parser_format}

python scripts/eval_extraction.py \
    -g ${folder_name} \
    -p ${folder_name} -w
