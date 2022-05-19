set -x
export CUDA_VISIBLE_DEVICES=2

python -u human_eval.py \
    --model_type "gpt2"\
    --repetition_penalty 1.1 \
    --model_name_or_path "model_100000"\
    --decode_strategy sampling \
    --batch_size 64 \
    --top_k 20 \
    --top_p 0.7 \
    --temperature 0.9 \
    --num_samples 100 \
    --max_dec_len 100 \
    --device "gpu" \
