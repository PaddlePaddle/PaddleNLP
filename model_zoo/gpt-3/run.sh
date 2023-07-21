
log_dir=mylog_single
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir=$log_dir --devices=0 --rank 0 tools/auto.py \
    -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.use_recompute=False \
    -o Global.global_batch_size=2 \
    -o Global.local_batch_size=2 \
    -o Global.micro_batch_size=2 \
    -o Engine.mix_precision.enable=False \
    -o Engine.max_steps=10 \
