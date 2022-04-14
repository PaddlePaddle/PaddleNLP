###
 # This script generates textual similarity predictions for texts contains rationales only and contains non-rationales only
###
export CUDA_VISIBLE_DEVICES=`python ./available_gpu.py --best 1`
export PYTHONPATH=./:$PYTHONPATH

BASE_MODEL=$1
INTER_MODE=$2
LANGUAGE=$3
TASK=similarity

for RATIONAL_TYPE in "rationale_text" "rationale_exclusive_text";
do
    if [[ $LANGUAGE == "en" ]]; then

        if [[ $BASE_MODEL == "roberta_base" ]]; then
            FROM_PRETRAIN=roberta-base
            CKPT=../task/${TASK}/pretrained_models/saved_model_${LANGUAGE}/roberta_base_20211109_205245/model_54000/model_state.pdparams
        elif [[ $BASE_MODEL == "roberta_large" ]]; then
            FROM_PRETRAIN=roberta-large
            CKPT=../task/${TASK}/pretrained_models/saved_model_${LANGUAGE}/roberta_large_20211109_205649/model_46000/model_state.pdparams
        elif [[ $BASE_MODEL == "lstm" ]]; then
            FROM_PRETRAIN=../task/${TASK}/skep_ernie_1.0_large_ch
            CKPT=../task/${TASK}/simnet/checkpoints_${LANGUAGE}/final.pdparams
        fi

    elif [[ $LANGUAGE == "ch" ]]; then

        if [[ $BASE_MODEL == "roberta_base" ]]; then
            FROM_PRETRAIN='roberta-wwm-ext'     
            CKPT=../task/${TASK}/pretrained_models/saved_model_${LANGUAGE}/roberta_base_20211018_104038/model_11400/model_state.pdparams
        elif [[ $BASE_MODEL == "roberta_large" ]]; then
            FROM_PRETRAIN='roberta-wwm-ext-large'       
            CKPT=../task/${TASK}/pretrained_models/saved_model_${LANGUAGE}/roberta_large_20211018_152833/model_22000/model_state.pdparams
        elif [[ $BASE_MODEL == "lstm" ]]; then
            FROM_PRETRAIN='skep_ernie_1.0_large_ch'
            CKPT=../task/${TASK}/simnet/checkpoints_${LANGUAGE}/final.pdparams
        fi
    fi

    OUTPUT=./prediction/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}/${RATIONAL_TYPE}/dev
    [ -d $OUTPUT ] || mkdir -p $OUTPUT
    set -x
    python3 similarity_pred.py  \
        --base_model $BASE_MODEL \
        --data_dir ./rationale/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}/${RATIONAL_TYPE}/dev \
        --output_dir $OUTPUT \
        --from_pretrained $FROM_PRETRAIN \
        --batch_size 1 \
        --max_seq_len 256 \
        --init_checkpoint $CKPT \
        --inter_mode  $INTER_MODE \
        --language $LANGUAGE
done