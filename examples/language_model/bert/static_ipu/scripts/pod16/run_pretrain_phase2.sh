#!/usr/bin/env bash

export RDMAV_FORK_SAFE=1
python3 run_pretrain.py \
        --input_files "path_to_phase2_hdf5_dataset" \
        --output_dir pretrain_384_model \
        --seq_len 384 \
        --hidden_size 768 \
        --vocab_size 30400 \
        --max_predictions_per_seq 56 \
        --max_position_embeddings 512 \
        --learning_rate 0.002828427125 \
        --weight_decay 1e-2 \
        --max_steps 2137 \
        --warmup_steps 274 \
        --logging_steps 10 \
        --seed 1984 \
        --beta1 0.9 \
        --beta2 0.999 \
        --num_hidden_layers 12 \
        --micro_batch_size 8 \
        --ipu_enable_fp16 True \
        --scale_loss 128 \
        --batches_per_step 1 \
        --num_replica 4 \
        --enable_grad_acc True \
        --grad_acc_factor 512 \
        --batch_size 16384 \
        --available_mem_proportion 0.28 \
        --ignore_index 0 \
        --enable_load_params True \
        --load_params_path "./pretrain_128_model/final_step_7038.pdparams" \
        --hidden_dropout_prob 0.1 \
        --attention_probs_dropout_prob 0.1 \
        --shuffle True \
        --wandb False \
        --enable_engine_caching False \
        --save_steps 500
