x# lora
```
python  finetune_generation.py \
    --model_name_or_path THUDM/chatglm2-6b  \
    --dataset_name_or_path /root/paddlejob/work/eb_data/hcg \
    --output_dir ./checkpoints \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 2 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --fp16 \
    --fp16_opt_level O2 \
    --do_ptq \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1  \
    --device gpu:3 \
    --do_ptq True
```
