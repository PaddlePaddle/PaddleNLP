###
 # This file contains script to run prediction of a specific baseline model and language on given input data
 # The result of this script will be used to evaluate the performance of the baseline model
###

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=./:$PYTHONPATH

LANGUAGE=ch                 # LANGUAGE choose in [en, ch]
BASE_MODEL=roberta_base     # BASE_MODEL choose in [roberta_base, roberta_large]

if [[ $LANGUAGE == "ch" ]]; then
    if [[ $BASE_MODEL == "roberta_base" ]]; then
        FROM_PRETRAIN=roberta-wwm-ext
        CKPT=models/roberta_base_DuReader-Checklist_20211022_095011/ckpt.bin    # 3epoch
        #CKPT=models/roberta_base_ch_20211220_202953/ckpt.bin #new fine_tune
    elif [[ $BASE_MODEL == "roberta_large" ]]; then
        FROM_PRETRAIN=roberta-wwm-ext-large
        # CKPT=models/ernie_large_DuReader-Checklist_20211007_163424/ckpt.bin     # 3 epoch F1: 63.465  EM: 52.832 
        # CKPT=models/ernie_large_DuReader-Checklist_20211009_115837/ckpt.bin     # 4 epoch F1: 63.323  EM: 52.920
        # CKPT=models/ernie_large_DuReader-Checklist_20211009_142730/ckpt.bin    # 3 epoch F1: 66.613    EM: 57.168
        CKPT=models/roberta_large_DuReader-Checklist_20211022_095359/ckpt.bin
        #CKPT=models/roberta_large_ch_20211220_203809/ckpt.bin #new fine_tune
    fi
elif [[ $LANGUAGE == "en" ]]; then
    if [[ $BASE_MODEL == "roberta_base" ]]; then
        FROM_PRETRAIN=roberta-base
        CKPT=models/roberta_base_squad2_20211113_104225/ckpt.bin
        #CKPT=models/roberta_base_en_20211221_201720/ckpt.bin #new fine_tune
    elif [[ $BASE_MODEL == "roberta_large" ]]; then
        FROM_PRETRAIN=roberta-large
        CKPT=models/roberta_large_squad2_20211113_111300/ckpt.bin
        #CKPT=models/roberta_large_en_20211223_114421/ckpt.bin #new fine_tune
    fi
fi

OUTPUT=./output/mrc_${LANGUAGE}.${BASE_MODEL}
[ -d $OUTPUT ] || mkdir -p $OUTPUT
set -x
python3 ./saliency_map/rc_prediction.py \
    --base_model $BASE_MODEL \
    --data_dir ../../data/mrc_${LANGUAGE} \
    --from_pretrained $FROM_PRETRAIN \
    --init_checkpoint $CKPT \
    --output_dir $OUTPUT \
    --n-samples 300 \
    --doc_stride 128 \
    --language $LANGUAGE \
    --max_seq_len 384 \
    --batch_size 32 \
    --epoch 2