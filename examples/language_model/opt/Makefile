
dist-finetune:
	python -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py \
		--model_name_or_path facebook/opt-125m \
		--num_train_epochs 4 \
		--learning_rate 3e-5 \
		--warmup_ratio 0.06 \
		--weight_decay 0.1 \
		--label_smoothing 0.1 \
		--save_steps 10000 \
		--logging_steps 1 \
		--eval_steps 4000 \
		--output_dir ./checkpoints/opt \
		--src_length 608 \
		--tgt_length 160 \
		--min_tgt_length 55 \
		--length_penalty 0.7 \
		--no_repeat_ngram_size 3 \
		--num_beams 5 \
		--select_topk True \
		--per_device_eval_batch_size 2 \
		--per_device_train_batch_size 2 \
		--max_grad_norm 1.0 \
		--lr_scheduler_type linear \
		--tensor_parallel_degree 4 \
		--fp16 \
		--recompute \
		--do_train \
		--do_eval

finetune:
	python finetune_generation.py \
		--model_name_or_path facebook/opt-125m \
		--num_train_epochs 4 \
		--learning_rate 3e-5 \
		--warmup_ratio 0.06 \
		--weight_decay 0.1 \
		--label_smoothing 0.1 \
		--save_steps 10000 \
		--logging_steps 1 \
		--eval_steps 40000 \
		--output_dir ./checkpoints/opt \
		--src_length 608 \
		--tgt_length 160 \
		--min_tgt_length 55 \
		--length_penalty 0.7 \
		--no_repeat_ngram_size 3 \
		--num_beams 5 \
		--select_topk True \
		--per_device_eval_batch_size 2 \
		--per_device_train_batch_size 2 \
		--max_grad_norm 1.0 \
		--lr_scheduler_type linear \
		--fp16 \
		--recompute \
		--do_train \
		--do_eval

.PYTHON: export
export:
	python export_generation.py \
		--model_name_or_path checkpoints/opt \
		--output_path inference/opt

predict:
	python predict_generation.py \
		--model_name_or_path checkpoints/opt

infer:
	make export
	python infer_generation.py \
		--model_dir inference/ \
		--model_prefix opt