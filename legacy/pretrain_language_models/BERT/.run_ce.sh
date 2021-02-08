export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1

BERT_BASE_PATH="chinese_L-12_H-768_A-12"
TASK_NAME='xnli'
DATA_PATH=data/xnli/XNLI-MT-1.0
CKPT_PATH=pretrain_model

train(){
python -u run_classifier.py --task_name ${TASK_NAME} \
                   --use_cuda true \
                   --do_train true \
                   --do_val false \
                   --do_test false \
                   --batch_size 8192 \
                   --in_tokens true \
                   --init_checkpoint pretrain_model/chinese_L-12_H-768_A-12/ \
                   --data_dir ${DATA_PATH} \
                   --vocab_path pretrain_model/chinese_L-12_H-768_A-12/vocab.txt \
                   --checkpoints ${CKPT_PATH} \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 25 \
                   --epoch 1 \
                   --max_seq_len 512 \
                   --bert_config_path pretrain_model/chinese_L-12_H-768_A-12/bert_config.json \
                   --learning_rate 1e-4 \
                   --skip_steps 10 \
                   --random_seed 100 \
                   --enable_ce \
                   --shuffle false
}

export CUDA_VISIBLE_DEVICES=0
train | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
train | python _ce.py
