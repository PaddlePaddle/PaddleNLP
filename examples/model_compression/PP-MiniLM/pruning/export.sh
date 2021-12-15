MODEL_PATH=$1
TASK_NAME=$2
python export_model.py --model_type ernie \
    --model_name_or_path ${MODEL_PATH}/${TASK_NAME}/0.75/best_model \
    --sub_model_output_dir ${MODEL_PATH}/${TASK_NAME}/0.75/sub/  \
    --static_sub_model ${MODEL_PATH}/${TASK_NAME}/0.75/sub_static/float  \
    --n_gpu 1 --width_mult 0.75
