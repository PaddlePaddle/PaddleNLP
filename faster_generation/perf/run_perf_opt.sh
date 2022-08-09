export CUDA_VISIBLE_DEVICES=3

for model_name in facebook/opt-125m facebook/opt-350m;
    do   
        for top_k in 1 4 8 16;
            do
                python opt_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=sampling \
                    --top_k=$top_k \
                    --top_p=0.4 \
                    --max_length=32 
                sleep 10s
                python opt_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=sampling \
                    --top_k=$top_k \
                    --top_p=0.4 \
                    --max_length=32 \
                    --use_fp16_decoding
                sleep 10s
            done
        python opt_perf.py \
            --model_name_or_path=$model_name \
            --decode_strategy=sampling \
            --top_k=0 \
            --top_p=0.4 \
            --max_length=32 
        sleep 10s
        python opt_perf.py \
            --model_name_or_path=$model_name \
            --decode_strategy=sampling \
            --top_k=0 \
            --top_p=0.4 \
            --max_length=32 \
            --use_fp16_decoding
        sleep 10s
    done
