export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

export FLAGS_call_stack_level=2
export GLOG_logtostderr=true
export GLOG_v=0

export FLAGS_control_flow_use_new_executor=1
export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.92
export CUDA_VISIBLE_DEVICES=1

python predictor.py \
    --model_name_or_path facebook/llama-13b \
    --dtype float16 \
    --src_length 1024 \
    --max_length 1024 \
    --output_file "infer.json" \
    --mode "dynamic" \
    --batch_size 1 \
    --block_attn \
    --inference_model