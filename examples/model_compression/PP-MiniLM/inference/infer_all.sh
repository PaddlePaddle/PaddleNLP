for task in afqmc tnews iflytek cmnli ocnli cluewsc2020 csl
do
    for bs in 4 8
    do
        for algo in abs_max avg hist mse
        do
            python infer.py --task_name ${task}  --model_path  ../quantization/${task}_quant_models/${algo}${bs}/int8  --int8 --use_trt --collect_shape
            python infer.py --task_name ${task}  --model_path  ../quantization/${task}_quant_models/${algo}${bs}/int8  --int8 --use_trt
            echo this is ${task}, ${algo}, ${bs}
        done
   done
done
