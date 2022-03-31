###
 # This file contains script to generate saliency map of all baseline models and languages on given input data
 # The result of this script will be used to evaluate the interpretive performance of the baseline model
### 
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=./:$PYTHONPATH

START_ID=0

for BASE_MODEL in "lstm" "roberta_base" "roberta_large";
do
    for INTER_MODE in "attention" "integrated_gradient" "lime";
    do
        for LANGUAGE in "ch" "en";
        do
            TASK=similarity_${LANGUAGE}
            DATA=../../data/${TASK}

            if [[ $LANGUAGE == "ch" ]]; then

                if [[ $BASE_MODEL == "roberta_base" ]]; then
                    FROM_PRETRAIN='roberta-wwm-ext'     
                    CKPT=pretrained_models/saved_model_ch/roberta_base_20211018_104038/model_11400/model_state.pdparams
                    #CKPT=pretrained_models/saved_model_ch/roberta_base_20211208_121026/model_12000/model_state.pdparams
                elif [[ $BASE_MODEL == "roberta_large" ]]; then
                    FROM_PRETRAIN='roberta-wwm-ext-large'       
                    CKPT=pretrained_models/saved_model_ch/roberta_large_20211018_152833/model_22000/model_state.pdparams
                    #CKPT=pretrained_models/saved_model_ch/roberta_large_20211208_131546/model_22000/model_state.pdparams
                elif [[ $BASE_MODEL == "lstm" ]]; then
                    FROM_PRETRAIN='data/skep_ernie_1.0_large_ch'
                    CKPT=simnet/checkpoints_ch/final.pdparams
                fi

            elif [[ $LANGUAGE == "en" ]]; then
                if [[ $BASE_MODEL == "roberta_base" ]]; then
                    FROM_PRETRAIN=roberta-base
                    CKPT=pretrained_models/saved_model_en/roberta_base_20211109_205245/model_54000/model_state.pdparams
                    #CKPT=pretrained_models/saved_model_en/roberta_base_20211208_121339/model_54000/model_state.pdparams
                elif [[ $BASE_MODEL == "roberta_large" ]]; then
                    FROM_PRETRAIN=roberta-large
                    CKPT=pretrained_models/saved_model_en/roberta_large_20211109_205649/model_46000/model_state.pdparams
                    #CKPT=pretrained_models/saved_model_en/roberta_large_20211208_131440/model_42000/model_state.pdparams
                elif [[ $BASE_MODEL == "lstm" ]]; then
                    FROM_PRETRAIN='data/skep_ernie_1.0_large_ch'
                    CKPT=simnet/checkpoints_en/final.pdparams
                fi
            fi

            OUTPUT=./output/$TASK.$BASE_MODEL
            [ -d $OUTPUT ] || mkdir -p $OUTPUT
            set -x
            if [[ ! -f ${OUTPUT}/interpret.${INTER_MODE} ]]; then
                python3 ./saliency_map/similarity_interpretable.py \
                    --base_model $BASE_MODEL \
                    --data_dir $DATA \
                    --from_pretrained $FROM_PRETRAIN \
                    --batch_size 1 \
                    --max_seq_len 256 \
                    --init_checkpoint $CKPT \
                    --inter_mode  $INTER_MODE \
                    --start_id $START_ID \
                    --output_dir $OUTPUT \
                    --n-samples 500 \
                    --language $LANGUAGE \
                    --eval $@
            fi
        done
    done
done
