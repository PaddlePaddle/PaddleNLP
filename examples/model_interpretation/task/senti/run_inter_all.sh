###
 # This file contains script to generate saliency map of all baseline models and languages on given input data
 # The result of this script will be used to evaluate the interpretive performance of the baseline model
###

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=./:$PYTHONPATH
START_ID=0
FROM_PRETRAIN='test'
VOCAB_PATH='test'

for BASE_MODEL in "lstm" "roberta_base" "roberta_large";
do
    for INTER_MODE in "attention" "integrated_gradient" "lime";
    do
        for LANGUAGE in "ch" "en";
        do
            TASK=senti_${LANGUAGE}
            DATA=../../data/${TASK}

            if [[ $LANGUAGE == "en" ]]; then

                if [[ $BASE_MODEL == "roberta_base" ]]; then
                    FROM_PRETRAIN='roberta-base'
                    CKPT=pretrained_models/saved_model_en/roberta_base_20211105_135732/model_10000/model_state.pdparams
                    #CKPT=pretrained_models/saved_model_en/roberta_base_20211206_164443/model_10000/model_state.pdparams
                elif [[ $BASE_MODEL == "roberta_large" ]]; then
                    FROM_PRETRAIN='roberta-large'
                    CKPT=pretrained_models/saved_model_en/roberta_large_20211105_160323/model_4000/model_state.pdparams
                    #CKPT=pretrained_models/saved_model_en/roberta_large_20211207_174631/model_4000/model_state.pdparams
                elif [[ $BASE_MODEL == "lstm" ]]; then
                    VOCAB_PATH='rnn/vocab.sst2_train'
                    CKPT=rnn/checkpoints_en/final.pdparams
                    #CKPT=rnn/checkpoints_en/final.pdparams
                fi

            elif [[ $LANGUAGE == "ch" ]]; then

                if [[ $BASE_MODEL == "roberta_base" ]]; then
                    FROM_PRETRAIN='roberta-wwm-ext'     
                    CKPT=pretrained_models/saved_model_ch/roberta_base/model_900/model_state.pdparams
                    #CKPT=pretrained_models/saved_model_ch/roberta_base_20211229_101252/model_900/model_state.pdparams
                elif [[ $BASE_MODEL == "roberta_large" ]]; then
                    FROM_PRETRAIN='roberta-wwm-ext-large'       
                    CKPT=pretrained_models/saved_model_ch/roberta_large_20211014_192021/model_900/model_state.pdparams
                    #CKPT=pretrained_models/saved_model_ch/roberta_large_20211229_105019/model_900/model_state.pdparams
                elif [[ $BASE_MODEL == "lstm" ]]; then
                    VOCAB_PATH='rnn/vocab.txt'
                    CKPT=rnn/checkpoints_ch/final.pdparams
                    #CKPT=rnn/checkpoints_ch/final.pdparams
                fi
            fi

            OUTPUT=./output/${TASK}.${BASE_MODEL}
            [ -d $OUTPUT ] || mkdir -p $OUTPUT
            set -x

            if [[ ! -f ${OUTPUT}/interpret.${INTER_MODE} ]]; then
                python3 ./saliency_map/sentiment_interpretable.py \
                    --language $LANGUAGE \
                    --base_model $BASE_MODEL \
                    --data_dir $DATA \
                    --vocab_path $VOCAB_PATH \
                    --from_pretrained $FROM_PRETRAIN \
                    --batch_size 1 \
                    --init_checkpoint $CKPT \
                    --inter_mode  $INTER_MODE\
                    --output_dir $OUTPUT \
                    --n-samples 200 \
                    --start_id $START_ID \
                    --eval $@
            fi
        done
    done
done