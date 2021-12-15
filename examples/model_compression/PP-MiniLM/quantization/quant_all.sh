MODEL_DIR=../pruning/pruned_models/

for task in AFQMC TNEWS IFLYTEK CMNLI OCNLI CLUEWSC2020 CSL

do

    python quant_post.py --task_name ${task} --input_dir ${MODEL_DIR}/${task}/0.75/sub_static

done
