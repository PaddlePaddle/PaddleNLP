CUDA_VISIBLE_DEVICES=$1 python run_eval.py \
--model_name THUDM/glm-10b \
--eval_path ./wikitext-103/wiki.test.tokens \
--overlapping_eval 256  \
--batch_size 8 \
--device gpu --seq_length 1024
