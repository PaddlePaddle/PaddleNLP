###
 # This script generates sentiment predictions for texts contains rationales only and contains non-rationales only
###

export CUDA_VISIBLE_DEVICES=`python ./available_gpu.py --best 1`
export PYTHONPATH=./:$PYTHONPATH

BASE_MODEL=$1
INTER_MODE=$2
LANGUAGE=$3
TASK=senti

FROM_PRETRAIN='test'
VOCAB_PATH='test'
for RATIONAL_TYPE in "rationale_text" "rationale_exclusive_text";
do
    if [[ $LANGUAGE == "en" ]]; then

        if [[ $BASE_MODEL == "roberta_base" ]]; then
            FROM_PRETRAIN=roberta-base
            CKPT=../task/${TASK}/pretrained_models/saved_model_en/roberta_base_20220318_185322/model_10000/model_state.pdparams
            #CKPT=../../../${TASK}/pretrained_models/saved_model_en/roberta_base_20211206_164443/model_10000/model_state.pdparams
        elif [[ $BASE_MODEL == "roberta_large" ]]; then
            FROM_PRETRAIN=roberta-large
            CKPT=../task/${TASK}/pretrained_models/saved_model_en/roberta_large_20220318_183813/model_4000/model_state.pdparams
            #CKPT=../../../${TASK}/pretrained_models/saved_model_en/roberta_large_20211207_174631/model_4000/model_state.pdparams
        elif [[ $BASE_MODEL == "lstm" ]]; then
            VOCAB_PATH=../task/${TASK}/rnn/vocab.sst2_train
            CKPT=../task/${TASK}/rnn/checkpoints_en/final.pdparams
        fi

    elif [[ $LANGUAGE == "ch" ]]; then

        if [[ $BASE_MODEL == "roberta_base" ]]; then
            FROM_PRETRAIN='roberta-wwm-ext'     
            CKPT=../task/${TASK}/pretrained_models/saved_model_ch/roberta_base_20220318_155933/model_900/model_state.pdparams
            #CKPT=../../../${TASK}/pretrained_models/saved_model_ch/roberta_base_20211206_180737/model_900/model_state.pdparams
        elif [[ $BASE_MODEL == "roberta_large" ]]; then
            FROM_PRETRAIN='roberta-wwm-ext-large'       
            CKPT=../task/${TASK}/pretrained_models/saved_model_ch/roberta_large_20220318_170123/model_900/model_state.pdparams
            #CKPT=../../../${TASK}/pretrained_models/saved_model_ch/roberta_large_20211207_143351/model_900/model_state.pdparams
        elif [[ $BASE_MODEL == "lstm" ]]; then
            VOCAB_PATH=../task/${TASK}/rnn/vocab.txt
            CKPT=../task/${TASK}/rnn/checkpoints_ch/final.pdparams
        fi
    fi

    OUTPUT=./prediction/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}/${RATIONAL_TYPE}/dev
    [ -d $OUTPUT ] || mkdir -p $OUTPUT
    set -x
    python3 ./sentiment_pred.py \
        --base_model $BASE_MODEL \
        --data_dir ./rationale/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}/${RATIONAL_TYPE}/dev \
        --output_dir $OUTPUT \
        --vocab_path $VOCAB_PATH \
        --from_pretrained $FROM_PRETRAIN \
        --batch_size 1 \
        --init_checkpoint $CKPT \
        --inter_mode  $INTER_MODE \
        --n-samples 200 \
        --language $LANGUAGE
done