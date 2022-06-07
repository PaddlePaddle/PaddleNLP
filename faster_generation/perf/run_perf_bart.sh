export CUDA_VISIBLE_DEVICES=3

for model_name in bart-base bart-large;  
    do   
        for top_k in 1 4 8 16;
            do
                python bart_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=sampling \
                    --num_beams=1 \
                    --top_k=$top_k \
                    --top_p=1 \
                    --max_length=32 
                sleep 10s
                python bart_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=sampling \
                    --num_beams=1 \
                    --top_k=$top_k \
                    --top_p=1 \
                    --max_length=32 \
                    --use_fp16_decoding
                sleep 10s
            done
        python bart_perf.py \
            --model_name_or_path=$model_name \
            --decode_strategy=sampling \
            --num_beams=1 \
            --top_k=0 \
            --top_p=0.4 \
            --max_length=32 
        sleep 10s
        python bart_perf.py \
            --model_name_or_path=$model_name \
            --decode_strategy=sampling \
            --num_beams=1 \
            --top_k=0 \
            --top_p=0.4 \
            --max_length=32 \
            --use_fp16_decoding
        sleep 10s
        for num_beams in 4 8 16;
            do
                python bart_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=beam_search \
                    --num_beams=$num_beams \
                    --top_k=1 \
                    --top_p=1 \
                    --max_length=32 
                sleep 10s
                python bart_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=beam_search \
                    --num_beams=$num_beams \
                    --top_k=1 \
                    --top_p=1 \
                    --max_length=32 \
                    --use_fp16_decoding
                sleep 10s
            done
    done