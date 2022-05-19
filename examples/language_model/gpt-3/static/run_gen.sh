set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit
unset CUDA_VISIBLE_DEVICES

rm -rf main_sharding*

task_name="gpt-generation"
rm -rf output/$task_name/log

python -u  -m paddle.distributed.fleet.launch \
    --gpus "0" \
    --log_dir "output/$task_name/log" run_generation.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-medium-en" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 1024 \
    --micro_batch_size 2 \
    --global_batch_size 2 \
    --max_dec_len 20 \
    --decoding_strategy 'topk_sampling' \
    --topp 0.9 \
    --save_inference_model_then_exist true \
    --device "gpu" 

