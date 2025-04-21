#!/bin/bash
# get filter shared_gpt dataset
if [ ! -f ./filtered_sharedgpt_short_3000.json ]; then
  python get_filter_shared_gpt.py --tokenizer_name $MODEL_NAME
fi

python benchmark_client.py \
  --dataset_path ./filtered_sharedgpt_short_3000.json \
  --backend $1 \
  --num_prompts 3000 \
  --warmup_round 1 \
  --concurrency 256 \
  --host localhost \
  --port 8110 \
  --dataset_name sharegpt \
  --max_dec_len 2048 \
  --output_file output.log