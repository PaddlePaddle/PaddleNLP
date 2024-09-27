#!/bin/bash
export FLAGS_enable_ixdnn_attn=True
export FLAGS_embedding_deterministic=1

# LoRA
python3 -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_lora_argument.json --output_dir ./kw/math/lora --logging_dir ./kw/math/lora --dataset_name_or_path ./SFT/eng_data/data_math
python3 -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_lora_argument.json --output_dir ./kw/code/lora --logging_dir ./kw/code/lora --dataset_name_or_path ./SFT/eng_data/data_code
python3 -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_lora_argument.json --output_dir ./kw/qa/lora --logging_dir ./kw/qa/lora --dataset_name_or_path ./SFT/eng_data/data_slim

# sft
python3 -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_sft_argument.json --output_dir ./kw/math/sft --logging_dir ./kw/math/sft --dataset_name_or_path ./SFT/eng_data/data_math --save_step 625 --do_eval False --eval_with_do_generation False --pipeline_parallel_degree 2 --evaluation_strategy no --ignore_save_lr_and_optim True
python3 tools/merge_tp_and_pp_params.py --model_name_or_path ./kw/math/sft/checkpoint-625 --pipeline_parallel_degree 2 --tensor_parallel_degree 8
python3 -u -m paddle.distributed.launch --gpus "8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_sft_argument.json --output_dir ./kw/math/sft --logging_dir ./kw/math/sft --dataset_name_or_path ./SFT/eng_data/data_math --do_train False

python3 -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_sft_argument.json --output_dir ./kw/code/sft --logging_dir ./kw/code/sft --dataset_name_or_path ./SFT/eng_data/data_code --save_step 593 --do_eval False --eval_with_do_generation False --pipeline_parallel_degree 2 --evaluation_strategy no --ignore_save_lr_and_optim True
python3 tools/merge_tp_and_pp_params.py --model_name_or_path ./kw/code/sft/checkpoint-593 --pipeline_parallel_degree 2 --tensor_parallel_degree 8
python3 -u -m paddle.distributed.launch --gpus "8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_sft_argument.json --output_dir ./kw/code/sft --logging_dir ./kw/code/sft --dataset_name_or_path ./SFT/eng_data/data_code --do_train False

python3 -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_sft_argument.json --output_dir ./kw/qa/sft --logging_dir ./kw/qa/sft --dataset_name_or_path ./SFT/eng_data/data_slim --save_step 625 --do_eval False --eval_with_do_generation False --pipeline_parallel_degree 2 --evaluation_strategy no --ignore_save_lr_and_optim True
python3 tools/merge_tp_and_pp_params.py --model_name_or_path ./kw/qa/sft/checkpoint-625 --pipeline_parallel_degree 2 --tensor_parallel_degree 8
python3 -u -m paddle.distributed.launch --gpus "8,9,10,11,12,13,14,15" run_finetune.py ./SFT/kw_sft_argument.json --output_dir ./kw/qa/sft --logging_dir ./kw/qa/sft --dataset_name_or_path ./SFT/eng_data/data_slim --do_train False