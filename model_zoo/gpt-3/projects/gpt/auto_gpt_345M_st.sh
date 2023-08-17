export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

log_dir=log/st_single
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir=$log_dir --devices=0 tools/auto.py \
    -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.use_recompute=True \
    -o Global.local_batch_size=1 \
    -o Global.micro_batch_size=1 \
    -o Distributed.dp_degree=1 \
    -o Distributed.mp_degree=1 \
    -o Distributed.pp_degree=1 \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Engine.mix_precision.enable=False \
    -o Engine.max_steps=100 \
    -o Engine.eval_freq=10 \
    -o Engine.logging_freq=1 \
    -o Engine.verbose=3 \
    -o Engine.save_load.output_dir=""