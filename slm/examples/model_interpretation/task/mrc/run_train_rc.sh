# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

###
 # This script is used to run fine-tunning of mrc roberta models.
### 

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=.:$PYTHONPATH

LANGUAGE=ch                    # LANGUAGE choose in [ch, en]
BASE_MODEL=roberta_base       # chooices [roberta_base, roberta_large]

[ -d "logs" ] || mkdir -p "logs"
set -x

if [[ $LANGUAGE == "ch" ]]; then
    if [[ $BASE_MODEL == "roberta_base" ]]; then
        FROM_PRETRAIN=roberta-wwm-ext
    elif [[ $BASE_MODEL == "roberta_large" ]]; then
        FROM_PRETRAIN=roberta-wwm-ext-large
    fi
    EPOCH=3
    BSZ=2
    LR=3e-5
    MAX_SEQLEN=512
    DATA=DuReader-Checklist
elif [[ $LANGUAGE == 'en' ]]; then
    if [[ $BASE_MODEL == "roberta_base" ]]; then
        FROM_PRETRAIN=roberta-base
    elif [[ $BASE_MODEL == "roberta_large" ]]; then
        FROM_PRETRAIN=roberta-large
    fi
    EPOCH=2
    BSZ=16
    LR=5e-6
    MAX_SEQLEN=384
    DATA=squad2
fi

timestamp=`date  +"%Y%m%d_%H%M%S"`
python3 saliency_map/rc_finetune.py \
    --train_data_dir ./data/$DATA/train/train.json \
    --dev_data_dir ./data/$DATA/dev/dev.json \
    --max_steps -1 \
    --from_pretrained $FROM_PRETRAIN \
    --epoch $EPOCH \
    --bsz $BSZ \
    --lr $LR \
    --max_seq_len $MAX_SEQLEN \
    --save_dir models/${BASE_MODEL}_${LANGUAGE}_${timestamp} \
    --language $LANGUAGE \
    --init_checkpoint models/${BASE_MODEL}_${LANGUAGE}_${timestamp}/ckpt.bin >> logs/log_${BASE_MODEL}_$timestamp 2>&1
    