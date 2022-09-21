python export_model.py \
    --model_name_or_path unimo-text-1.0-summary \
    --decoding_strategy beam_search \
    --inference_model_dir ./inference_model \
    --max_out_len 30 \