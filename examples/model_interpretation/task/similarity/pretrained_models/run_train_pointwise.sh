###
 # This script is used to finetune pretrained models
###

export CUDA_VISIBLE_DEVICES=7

LANGUAGE="ch"               # ['ch', 'en']
BASE_MODEL=roberta_large     # [roberta_base, roberta_large]
timestamp=`date  +"%Y%m%d_%H%M%S"`

if [[ $LANGUAGE == "ch" ]]; then
    LEARNING_RATE=3e-5
    MAX_SEQ_LENGTH=256
elif [[ $LANGUAGE == "en" ]]; then
    LEARNING_RATE=5e-6
    MAX_SEQ_LENGTH=128
fi

[ -d "logs" ] || mkdir -p "logs"
set -x

python3 ./train_pointwise.py  \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --batch_size 32 \
    --epochs 5 \
    --save_step 1000 \
    --warmup_proportion 0.1 \
    --base_model $BASE_MODEL \
    --language $LANGUAGE \
    --save_dir saved_model_${LANGUAGE}/${BASE_MODEL}_${timestamp} >> logs/log_${BASE_MODEL}_${timestamp}
    
