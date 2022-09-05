GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

for model_name in Salesforce/codegen-350M-mono Salesforce/codegen-2B-mono Salesforce/codegen-6B-mono; 
    do   
        for top_k in 1 4 8 16;
            do
                for input_len in 60;
                    do
                        for generate_len in 20;
                            do
                                for perf_type in pd pd_faster_fp32 pd_faster_fp16 hf;
                                    do 
                                        echo model_name: $model_name, perf_type: $perf_type, top_k: $top_k, top_p: 1.0, input_len: $input_len, generate_len: $generate_len
                                        python codegen_perf.py \
                                            --model_name_or_path=$model_name \
                                            --perf_type=$perf_type \
                                            --top_k=$top_k \
                                            --top_p=1.0 \
                                            --input_len=$input_len \
                                            --generate_len=$generate_len \
                                            --gpu_id ${GPU_ID}
                                        sleep 3s
                                    done
                            done
                    done
            done
        for top_p in 0.4;
            do
                for input_len in 60;
                    do
                        for generate_len in 20;
                            do
                                for perf_type in pd pd_faster_fp32 pd_faster_fp16 hf;
                                    do 
                                        echo model_name: $model_name, perf_type: $perf_type, top_k: 0, top_p: $top_p, input_len: $input_len, generate_len: $generate_len
                                        python codegen_perf.py \
                                            --model_name_or_path=$model_name \
                                            --perf_type=$perf_type \
                                            --top_k=0 \
                                            --top_p=$top_p \
                                            --input_len=$input_len \
                                            --generate_len=$generate_len \
                                            --gpu_id ${GPU_ID}
                                        sleep 3s
                                    done
                            done
                    done
            done
    done