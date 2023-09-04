export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

export FLAGS_control_flow_use_new_executor=1
export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.92

python predictor.py \
    --model_name_or_path ./llama-13b-inference_model_fp16 \
    --dtype float16 \
    --src_length 300 \
    --max_length 400 \
    --output_file "infer.json" \
    --mode "static" \
    --batch_size 1 \
    --benchmark \
    --inference_model