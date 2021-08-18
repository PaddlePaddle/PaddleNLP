
set -eux

export PYTHONPATH="/home/tianxin04/develop/PaddleNLP/"
PYTHON_BIN="/usr/local/bin/python3.7"

#train_set="./unsupervised_corpus.tsv"
#train_set="./lcqmc/train_1p6w.txt"


model="ernie1.0"
gpu="0,1"
tasks=(LCQMC BQ ATEC PAWSX STS-B)
tasks=(LCQMC)
lrs=(1E-5 5E-5)
dropouts=(0.1 0.3)

function train(){

	local gpu=$1
	local lr=$2
	local dropout=$3

	export CUDA_VISIBLE_DEVICES=${gpu}

	#${PYTHON_BIN} -u -m paddle.distributed.launch --gpus ${gpu} \
	${PYTHON_BIN} -u \
		train.py \
		--device gpu \
		--save_dir ./checkpoints/ \
		--batch_size 64 \
		--learning_rate ${lr} \
		--epochs 1 \
		--output_emb_size -1 \
		--save_steps 50000 \
		--eval_steps 100 \
		--max_seq_length 64 \
		--infer_with_fc_pooler \
		--margin 0.0 \
		--scale 20 \
		--dropout ${dropout} \
		--train_set_file ${train_set} \
		--test_set_file ${test_set} > ./log/${strategy_name} 2>&1
		#--output_emb_size 256 \
}

for task in ${tasks[@]}; do
	train_set="./senteval_cn/${task}/train.txt"
	test_set="./senteval_cn/${task}/dev.tsv"
	for lr in ${lrs[@]}; do
	for dropout in ${dropouts[@]}; do
		#strategy_name=${model}_${task}_lr${lr}_dropout${dropout}_infer_with_no_fc
		strategy_name=${model}_${task}_lr${lr}_dropout${dropout}_infer_with_fc
		echo ${strategy_name}
		train ${gpu} ${lr} ${dropout}
	done
	done
done
