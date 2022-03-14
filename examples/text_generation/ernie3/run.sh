export CUDA_VISIBLE_DEVICES=0
python3.7 infer.py \
--model_name_or_path ernie3-10b \
--topk 4 \
--topp 0.0 \
--temperature 0.9 \
--repetition_penalty 1.1 \
--min_out_len 10 \
--max_out_len 20 \
--decoding_strategy sampling
