###
 # This script is used to train lstm models
###

unset CUDA_VISIBLE_DEVICES
LANGUAGE=en

if [[ $LANGUAGE == "ch" ]]; then
    VOCAB_PATH=vocab.char
elif [[ $LANGUAGE == "en" ]]; then
    VOCAB_PATH=vocab_QQP
fi

python -m paddle.distributed.launch --gpus "5" train.py \
    --device=gpu \
    --lr=4e-4 \
    --batch_size=64 \
    --epochs=12 \
    --vocab_path=$VOCAB_PATH   \
    --language=$LANGUAGE \
    --save_dir="./checkpoints_"${LANGUAGE}
