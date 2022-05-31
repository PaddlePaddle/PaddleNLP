export CUDA_VISIBLE_DEVICES=0

python3 ./run_docvqa.py \
    --model_name_or_path "layoutxlm-base-uncased" \
    --max_seq_len 512 \
    --train_file "data/train.json" \
    --init_checkpoint "checkpoints/base_model" \
	--do_train true \
    --num_train_epochs 50 \
    --eval_steps 24000 \
    --save_steps 40 \
    --output_dir "output" \
    --save_path "data/decode_res.json" \
    --learning_rate 3e-5 \
    --warmup_steps 40 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --seed 2048
