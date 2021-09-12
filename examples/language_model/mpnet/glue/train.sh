# ["cola","sst-2","mrpc","sts-b","qqp","mnli", "rte", "qnli"]
unset CUDA_VISIBLE_DEVICES
# QQP
# 运行训练
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --task_name qqp \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_steps 5666 \
    --max_steps 113272 \
    --logging_steps 500 \
    --save_steps 2000 \
    --seed 42 \
    --output_dir qqp/ \
    --device gpu

# COLA
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --task_name cola \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_proportion 0.06 \
    --num_train_epochs 10 \
    --logging_steps 200 \
    --save_steps 200 \
    --seed 42 \
    --output_dir cola \
    --device gpu

# QNLI
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --task_name qnli \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_proportion 0.06 \
    --num_train_epochs 10 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --seed 42 \
    --output_dir qnli \
    --device gpu
  
# SST2
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --task_name sst-2 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_proportion 0.06 \
    --num_train_epochs 10 \
    --logging_steps 400 \
    --save_steps 400 \
    --seed 42 \
    --output_dir sst-2 \
    --device gpu


############################################################################################################################################
# 先训练这个模型，之后需要使用这个权重！(RTE，MRPC和STS-B用了MNLI做初始化，与roberta一致)
# MNLI
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --task_name mnli \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_proportion 0.06 \
    --num_train_epochs 10 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --seed 42 \
    --output_dir mnli \
    --device gpu

########################################################
# RTE
export MNLI_BEST_CKPT=/path/to/mnli/best/ckpt
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path $MNLI_BEST_CKPT \
    --task_name rte \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_proportion 0.06 \
    --num_train_epochs 13 \
    --logging_steps 100 \
    --save_steps 100 \
    --seed 42 \
    --output_dir rte \
    --device gpu
    
############################################################
# MRPC
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path $MNLI_BEST_CKPT \
    --task_name mrpc \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_proportion 0.06 \
    --num_train_epochs 10 \
    --logging_steps 100 \
    --save_steps 100 \
    --seed 42 \
    --output_dir mrpc \
    --device gpu  
  
############################################################
# STSB
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path $MNLI_BEST_CKPT \
    --task_name rte \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_proportion 0.06 \
    --num_train_epochs 10 \
    --logging_steps 100 \
    --save_steps 100 \
    --seed 42 \
    --output_dir rte \
    --device gpu
  
############################################################

