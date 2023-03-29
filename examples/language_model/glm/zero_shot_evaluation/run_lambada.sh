CUDA_VISIBLE_DEVICES=$1 python run_eval.py \
--model_name THUDM/glm-10b \
--eval_path ./lambada_test.jsonl \
--cloze_eval \
--batch_size 8 \
--device gpu

