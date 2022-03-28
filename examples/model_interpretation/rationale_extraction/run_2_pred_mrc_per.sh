###
 # This script generates mrc predictions for texts contains rationales only and contains non-rationales only
###
export CUDA_VISIBLE_DEVICES=`python ./available_gpu.py --best 1`
export PYTHONPATH=./:$PYTHONPATH

BASE_MODEL=$1
INTER_MODE=$2
LANGUAGE=$3
TASK=mrc

for RATIONAL_TYPE in "rationale_text" "rationale_exclusive_text";
do
    if [[ $LANGUAGE == "ch" ]]; then
        if [[ $BASE_MODEL == "roberta_base" ]]; then
            FROM_PRETRAIN=roberta-wwm-ext
            CKPT=../task/${TASK}/models/roberta_base_DuReader-Checklist_20211022_095011/ckpt.bin    # 3 epoch
            
        elif [[ $BASE_MODEL == "roberta_large" ]]; then
            FROM_PRETRAIN=roberta-wwm-ext-large
            CKPT=../task/${TASK}/models/roberta_large_DuReader-Checklist_20211022_095359/ckpt.bin   # 3 epoch
        fi
    elif [[ $LANGUAGE == "en" ]]; then
        if [[ $BASE_MODEL == "roberta_base" ]]; then
            FROM_PRETRAIN=roberta-base
            CKPT=../task/${TASK}/models/roberta_base_squad2_20211113_104225/ckpt.bin
            
        elif [[ $BASE_MODEL == "roberta_large" ]]; then
            FROM_PRETRAIN=roberta-large
            CKPT=../task/${TASK}/models/roberta_large_squad2_20211113_111300/ckpt.bin
        fi
    fi

    OUTPUT=./prediction/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}/${RATIONAL_TYPE}/dev
    [ -d $OUTPUT ] || mkdir -p $OUTPUT
    set -x
    python3 ./mrc_pred.py  \
        --input_data ../data/${TASK}_${LANGUAGE} \
        --base_model $BASE_MODEL \
        --data_dir ./rationale/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}/${RATIONAL_TYPE}/dev \
        --output_dir $OUTPUT \
        --from_pretrained $FROM_PRETRAIN \
        --batch_size 1 \
        --init_checkpoint $CKPT \
        --n-samples 300 \
        --doc_stride 128 \
        --language $LANGUAGE
done
