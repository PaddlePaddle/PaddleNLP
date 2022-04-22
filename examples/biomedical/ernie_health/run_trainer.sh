#set -x
unset CUDA_VISIBLE_DEVICES

task_name="eheath-pre"
rm -rf output/$task_name/log

PYTHONPATH=../../../  python -u -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7  \
    run_pretrain_trainer.py \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_length 512 \
    --gradient_accumulation_steps 1\
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 0.001 \
    --max_steps 1000000 \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --dataloader_num_workers 4 \
    --device "gpu"\
    --fp16  \
    --fp16_opt_level "O2"  \
    --do_train \
    --report_to "visualdl" \
    --save_total_limit 3 \
    --attention_probs_dropout_prob 0.0 \
    --hidden_dropout_prob 0.0\
    --overwrite_output_dir