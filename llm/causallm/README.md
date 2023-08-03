# sft
```
export MODEL='THUDM/chatglm-6b'
export DATA='data'
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py \
    --model_name_or_path $MODEL  \
    --dataset_name_or_path $DATA \
    --output_dir ./checkpoints \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --fp16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1 \
    --tensor_parallel_degree 4 \
    --eval_accumulation_steps 16
```
# lora
```
export MODEL='THUDM/chatglm-6b'
export DATA='data'
python  finetune_generation.py \
    --model_name_or_path $MODEL  \
    --dataset_name_or_path $DATA \
    --output_dir ./checkpoints \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --fp16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1  \
    --lora True \
    --eval_accumulation_steps 16
```
# prefix_tuning
```
export MODEL='THUDM/chatglm-6b'
export DATA='data'
python  finetune_generation.py \
    --model_name_or_path $MODEL \
    --dataset_name_or_path $DATA \
    --output_dir ./checkpoints \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 3e-2 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --fp16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1  \
    --prefix_tuning True \
    --eval_accumulation_steps 16
```
# server
```
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" server.py \
    --model_name_or_path facebook/llama-7b \
    --port 8010 \
    --flask_port 8011 \
    --src_length 100 \
    --dtype "float16" \ #非lora、pt，需要传入dtype
    --lora_path ./checkpoint  \ # 如果加载lora权重
    --gpt True # 如果是gpt模型选择True
```
