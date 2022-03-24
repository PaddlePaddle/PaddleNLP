#!/usr/bin/env bash

python3 run_squad.py \
        --output_dir squad_model \
        --task "SQUAD" \
        --is_training False \
        --seq_len 384 \
        --hidden_size 768 \
        --vocab_size 30400 \
        --max_predictions_per_seq 56 \
        --max_position_embeddings 512 \
        --learning_rate 5.6e-05 \
        --weight_decay 1e-2 \
        --epochs 4 \
        --warmup_steps 52 \
        --logging_steps 10 \
        --seed 1984 \
        --beta1 0.9 \
        --beta2 0.999 \
        --num_hidden_layers 12 \
        --micro_batch_size 2 \
        --ipu_enable_fp16 True \
        --scale_loss 256 \
        --optimizer_state_offchip False \
        --batches_per_step 4 \
        --num_replica 4 \
        --num_ipus 2 \
        --enable_grad_acc False \
        --grad_acc_factor 1 \
        --available_mem_proportion 0.40 \
        --ignore_index 0 \
        --hidden_dropout_prob 0.0 \
        --attention_probs_dropout_prob 0.0 \
        --shuffle False \
        --wandb False \
        --enable_engine_caching False \
        --enable_load_params True \
        --load_params_path "squad_model/Final_model.pdparams"
