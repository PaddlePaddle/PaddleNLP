set -eux

export BATCH_SIZE=8
export CKPT=./checkpoints/model_90000.pdparams
export DATASET_FILE=./data/dev_data.json

CUDA_VISIBLE_DEVICES=0 python run_duie.py \
                   --do_predict \
                   --init_checkpoint $CKPT \
                   --predict_data_file $DATASET_FILE \
                   --max_seq_length 512 \
                   --batch_size $BATCH_SIZE

