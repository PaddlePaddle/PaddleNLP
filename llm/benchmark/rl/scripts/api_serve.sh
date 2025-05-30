output_dir="your_output_dir"

python api_serve.py \
    --openai_urls "your_url1" "your_url2"\
    --api_keys "key1" "key2" \
    --model "Qwen2.5-7B-Instruct-1M" \
    --tokenizer "Qwen/Qwen2.5-7B-Instruct-1M" \
    --use_fastdeploy true \
    --input_file your_file \
    --output_dir ${output_dir} \
    --rollout_input_batch_size 8 \
    --rollout_n 8 \
    --top_p 1.0 \
    --temperature 0.7 \
    --max_dec_len 8192 \
    --limit_rows 512