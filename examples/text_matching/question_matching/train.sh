

unset CUDA_VISIBLE_DEVICES

PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="/ssd2/tianxin04/PaddleNLP/"

train_set="data_v1/train/ALL/train"
dev_set="data_v1/train/ALL/dev"


rdrop_coef=0.0
bs=32
lr=2E-5

strategy=rdrop${rdrop_coef}_bs${bs}_lr${lr}

${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "0" train.py \
		--train_set ${train_set} \
		--dev_set ${dev_set} \
		--test_set "" \
        --device gpu \
        --save_dir ./checkpoints_${strategy} \
		--rdrop_coef ${rdrop_coef} \
        --train_batch_size ${bs} \
        --learning_rate ${lr} > ${strategy}.log 2>&1 &
