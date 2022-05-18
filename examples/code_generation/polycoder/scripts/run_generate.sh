set -x
export CUDA_VISIBLE_DEVICES=0

python -u run_generation.py \
    --model_type "gpt2"\
    --model_name_or_path "model_100000"\
    --decode_strategy sampling \
    --repetition_penalty 1.2 \
    --top_k 10 \
    --temperature 0.5 \
    --num_return_sequences 10 \
    --max_dec_len 512 \
    --device "gpu" \
