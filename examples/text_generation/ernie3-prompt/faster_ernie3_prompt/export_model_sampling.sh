python3.7 ernie3_prompt_export_model_sample.py \
--model_name_or_path models \
--decoding_strategy sampling \
--inference_model_dir infer_model_fp16_k8_p09_seqlen128_20221205 \
--topk 5 \
--topp 1.0 \
--temperature 1.0 \
--max_out_len 128 \
--use_fp16_decoding \