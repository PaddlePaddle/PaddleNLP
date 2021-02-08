export FLAGS_fraction_of_gpu_memory_to_use=0.5
export FLAGS_eager_delete_tensor_gb=0.0 
export FLAGS_fast_eager_deletion_mode=1
export CUDA_VISIBLE_DEVICES=0

python train.py \
--traindata_dir data/train \
--model_save_dir model \
--use_gpu 1 \
--corpus_type_list train \
--corpus_proportion_list 1 \
--num_iterations 200000 \
--pretrain_elmo_model_path ${ELMo_MODEL_PATH} \
--testdata_dir data/dev $@ \
