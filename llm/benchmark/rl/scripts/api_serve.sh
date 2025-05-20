source /root/paddlejob/workspace/env_run/paddlenlp_rl/paddle.sh

INPUT_FILE=
OUTPUT_DIR=

python api_serve.py \
    --openai_urls "your urls" \
    --api_keys "NONE" \
    --model "Qwen2.5-7B-Instruct-1M" \
    --tokenizer "Qwen/Qwen2.5-7B-Instruct-1M" \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --rollout_input_batch_size 8 \
    --rollout_n 8 \
    --top_p 1.0 \
    --temperature 0.7 \
    --max_prompt_length 512 \
    --max_response_length 8192 \
    --limit_rows 512