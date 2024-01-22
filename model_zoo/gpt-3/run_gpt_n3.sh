export NCCL_IB_GID_INDEX=3

export log_dir=log_new

#rm -rf $log_dir
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    --auto_cluster_config true \
    --master=10.95.147.146:8091 \
    --nnodes=3 ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_full_auto_parallel_n3.yaml

