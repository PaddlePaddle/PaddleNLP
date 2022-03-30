###
 # This script is used to finetune pretrained models
###

export CUDA_VISIBLE_DEVICES=5

LANGUAGE=en
BASE_MODEL=roberta_base        # [roberta_base, roberta_large]
timestamp=`date  +"%Y%m%d_%H%M%S"`

if [[ $LANGUAGE == "ch" ]]; then
    LEARNING_RATE=2e-5
    MAX_SEQ_LENGTH=128
elif [[ $LANGUAGE == "en" ]]; then
    LEARNING_RATE=5e-6
    MAX_SEQ_LENGTH=512
fi

[ -d "logs" ] || mkdir -p "logs"
set -x

python3 ./train.py  \
    --learning_rate ${LEARNING_RATE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --batch_size 32 \
    --epochs 5 \
    --base_model $BASE_MODEL \
    --save_dir saved_model_${LANGUAGE}/${BASE_MODEL}_${timestamp} \
    --language $LANGUAGE >> logs/log_${BASE_MODEL}_${timestamp}
    
