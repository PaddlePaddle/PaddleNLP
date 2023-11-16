export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_new_executor_micro_batching=True

log_dir=amptrue/log_static_1.3B
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir=$log_dir --devices=4 tasks/gpt/train_static.py \
    -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
    -o Global.local_batch_size=1 \
    -o Global.micro_batch_size=1 \
    -o Global.enable_partial_send_recv=False \
    -o Engine.max_steps=1000 \
    -o Engine.logging_freq=1 \
    -o Engine.eval_freq=10000 \
    -o Engine.mix_precision.enable=True \
    -o Engine.save_load.save_steps=10000 \
    -o Model.use_recompute=False \
    -o Model.hidden_size=2048 \
    -o Model.num_layers=24 \
    -o Model.num_attention_heads=16 \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Optimizer.name=AdamW \
    -o Distributed.dp_degree=1 \
    -o Distributed.mp_degree=1 \
    -o Distributed.pp_degree=1 \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Distributed.sharding.reduce_overlap=False \
    -o Profiler_pretrain.memory_stats=True \
    -o Engine.verbose=3
