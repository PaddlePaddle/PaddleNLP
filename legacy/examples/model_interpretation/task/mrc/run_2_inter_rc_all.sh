###
 # This file contains script to generate saliency map of all baseline models and languages on given input data
 # The result of this script will be used to evaluate the interpretive performance of the baseline model
###

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=./:$PYTHONPATH

START=0
TASK=mrc
for BASE_MODEL in "roberta_base" "roberta_large";
do
    for INTER_MODE in "attention" "integrated_gradient";
    do
        for LANGUAGE in "ch" "en";
        do
            if [[ $LANGUAGE == "ch" ]]; then
                if [[ $BASE_MODEL == "roberta_base" ]]; then
                    FROM_PRETRAIN=roberta-wwm-ext
                    CKPT=models/roberta_base_DuReader-Checklist_20211022_095011/ckpt.bin    # 3 epoch
                    
                elif [[ $BASE_MODEL == "roberta_large" ]]; then
                    FROM_PRETRAIN=roberta-wwm-ext-large
                    CKPT=models/roberta_large_DuReader-Checklist_20211022_095359/ckpt.bin   # 3 epoch
                fi
            elif [[ $LANGUAGE == "en" ]]; then
                if [[ $BASE_MODEL == "roberta_base" ]]; then
                    FROM_PRETRAIN=roberta-base
                    CKPT=models/roberta_base_squad2_20211113_104225/ckpt.bin
                    
                elif [[ $BASE_MODEL == "roberta_large" ]]; then
                    FROM_PRETRAIN=roberta-large
                    CKPT=models/roberta_large_squad2_20211113_111300/ckpt.bin
                fi
            fi


            OUTPUT=./output/mrc_${LANGUAGE}.${BASE_MODEL}
            [ -d $OUTPUT ] || mkdir -p $OUTPUT
            set -x

            if [[ ! -f ${OUTPUT}/interpret.${INTER_MODE} ]]; then
                python3 ./saliency_map/rc_interpretable.py \
                    --ans_path ./output/${TASK}_${LANGUAGE}.${BASE_MODEL}/predict_ans\
                    --ans_idx_path ./output/${TASK}_${LANGUAGE}.${BASE_MODEL}/predict_feature_index\
                    --base_model $BASE_MODEL \
                    --data_dir ../../data/mrc_${LANGUAGE} \
                    --from_pretrained $FROM_PRETRAIN \
                    --batch_size 1 \
                    --init_checkpoint $CKPT \
                    --inter_mode  $INTER_MODE\
                    --output_dir $OUTPUT \
                    --n-samples 300 \
                    --doc_stride 128 \
                    --start_step $START \
                    --language $LANGUAGE\
                    --num_classes 2
            fi
        done
    done
done