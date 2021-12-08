export CUDA_VISIBLE_DEVICES=3

for model_name in gpt2-en gpt2-medium-en gpt2-large-en;  
    do   
        for top_k in 1 4 8 16;
            do
                python gpt_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=sampling \
                    --top_k=$top_k \
                    --top_p=1 \
                    --max_length=32 
                sleep 10s
                python gpt_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=sampling \
                    --top_k=$top_k \
                    --top_p=1 \
                    --max_length=32 \
                    --use_fp16_decoding
                sleep 10s
            done
        python gpt_perf.py \
            --model_name_or_path=$model_name \
            --decode_strategy=sampling \
            --top_k=0 \
            --top_p=0.4 \
            --max_length=32 
        sleep 10s
        python gpt_perf.py \
            --model_name_or_path=$model_name \
            --decode_strategy=sampling \
            --top_k=0 \
            --top_p=0.4 \
            --max_length=32 \
            --use_fp16_decoding
        sleep 10s
    done