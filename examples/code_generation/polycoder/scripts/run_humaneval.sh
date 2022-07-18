set -x
export CUDA_VISIBLE_DEVICES=2

python -u human_eval.py \
    --model_type "gpt2"\
    --repetition_penalty 1.0 \
    --model_name_or_path "model_1000000"\
    --merge_file "./data_tools/code-merges.txt" \
    --vocab_file "./data_tools/code-vocab.json" \
    --decode_strategy sampling \
    --batch_size 64 \
    --top_k 10 \
    --top_p 1.0 \
    --temperature 0.4 \
    --num_samples 100 \
    --max_dec_len 256 \
    --device "gpu" \
