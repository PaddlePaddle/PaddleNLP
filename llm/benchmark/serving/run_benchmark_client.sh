#!/bin/bash
# get filter shared_gpt dataset
# if [ ! -f ./filtered_sharedgpt_short_3000.json ]; then
#   python get_filter_shared_gpt.py --tokenizer_name $MODEL_NAME
# fi

python benchmark_client.py \
  --dataset_path ./filtered_sharedgpt_1w_input_228_output_195_real.json \
  --backend paddle \
  --num_prompts 1000 \
  --warmup_round 0 \
  --concurrency 40 \
  --host localhost \
  --port 9965 \
  --dataset_name sharegpt \
  --max_dec_len 2048 \
  --output_file output_dec_2048.log