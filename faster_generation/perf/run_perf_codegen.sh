export CUDA_VISIBLE_DEVICES=2

for model_name in Salesforce/codegen-350M-mono Salesforce/codegen-2B-mono; 
    do   
        for batch_size in 1 4 8 16;
            do
                for input_len in 60;
                    do
                        for generate_len in 20;
                            do
                                for perf_typle in pd_faster_fp32 pd_faster_fp16 hf;
                                    do 
                                        echo model_name: $model_name, perf_typle: $perf_typle, batch_size: $batch_size, input_len: $input_len, generate_len: $generate_len
                                        python codegen_perf.py \
                                            --model_name_or_path=$model_name \
                                            --perf_typle=$perf_typle \
                                            --batch_size=$batch_size \
                                            --input_len=$input_len \
                                            --generate_len=$generate_len \

                                        sleep 3s
                                    done
                            done
                    done
            done
    done