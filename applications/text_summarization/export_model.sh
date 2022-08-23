python export_model_unimo_text.py \
    --model_name_or_path ./checkpoint \
    --inference_model_dir ./export_checkpoint \
    --max_out_len 64 \
    --use_fp16_decoding