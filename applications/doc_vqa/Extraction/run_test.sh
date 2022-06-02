export CUDA_VISIBLE_DEVICES=0

QUESTION=$1

python3 change_to_mrc.py ${QUESTION}

python3  ./run_docvqa.py \
    --model_name_or_path "layoutxlm-base-uncased" \
    --max_seq_len 512 \
    --do_test true \
	--test_file "data/demo_test.json" \
	--num_train_epochs 100 \
    --eval_steps 6000 \
    --save_steps 6000 \
    --output_dir "output/" \
    --save_path "data/decode_res.json" \
	--init_checkpoint "./checkpoints/layoutxlm/" \
    --learning_rate 3e-5 \
    --warmup_steps 12000 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 1 \
    --seed 2048

python3 view.py
