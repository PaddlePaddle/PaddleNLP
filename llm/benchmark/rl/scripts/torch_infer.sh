INPUT_FILE=
OUTPUT_DIR=

python torch_infer.py \
    --actor_model_name_or_path /root/paddlejob/workspace/env_run/paddlenlp_rl/model/torch/Qwen2.5-7B-Instruct-1M \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --rollout_input_batch_size 8 \
    --rollout_n 8 \
    --tensor_parallel_degree 2 \
    --limit_rows 512 \
    --max_prompt_length 512 \
    --max_response_length 8192