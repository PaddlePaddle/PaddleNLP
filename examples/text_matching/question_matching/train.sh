

unset CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES="0,1,2,3"

PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="/ssd2/tianxin04/PaddleNLP/"

train_set="data_v1/train/ALL/train"
dev_set="data_v1/train/ALL/dev"


rdrop_coefs=(0)
#rdrop_coefs=(0.2)

#pretrained_model="ernie1p0"
pretrained_model="bert-base-chinese"

#rdrop_coefs=(0.1 0.3 0.5 1.0 5 10)
bs=32
lr=2E-5
gpus="0,1,2,3"


for rdrop_coef in ${rdrop_coefs[@]}; do
	strategy=${pretrained_model}_rdrop${rdrop_coef}_bs${bs}_lr${lr}
	echo ${strategy}
	#${PYTHON_BIN} -u train.py \
	${PYTHON_BIN} -u -m paddle.distributed.launch --gpus ${gpus} train.py \
			--train_set ${train_set} \
			--dev_set ${dev_set} \
			--device gpu \
			--eval_step 100 \
			--save_dir ./checkpoints_${strategy} \
			--rdrop_coef ${rdrop_coef} \
			--train_batch_size ${bs} \
			--learning_rate ${lr} > ${strategy}.log 2>&1
done
