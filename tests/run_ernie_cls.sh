batch_size=32
epoch=1
device=gpu
gpu_list="2,3"
save_dir=./ernie_text_cls_ckpt
static_model_dir=./ernie_text_cls_export
params_path=./ernie_text_cls_ckpt/model_100/model_state.pdparams
static_output_path=${static_model_dir}/static_graph_params
static_model_file=${static_model_dir}/static_graph_params.pdmodel
static_params_file=${static_model_dir}/static_graph_params.pdiparams
train_script=../examples/text_classification/pretrained_models/train.py
export_script=../examples/text_classification/pretrained_models/export_model.py
predict_script=../examples/text_classification/pretrained_models/deploy/python/predict.py
distributed_log_dir=./ernie_text_cls_log
echo "check and create directory"
if [ ! -d ${static_model_dir} ]; then
    mkdir ${static_model_dir}
    echo "create dir ${static_model_dir}."
else
    echo "dir ${static_model_dir} has already existed."
fi

process_name=${1}

if [ ${process_name} == ernie_text_cls_train ]; then
    echo -e "\nstart ernie text classfication training."
    unset CUDA_VISIBLE_DEVICES
    python -m paddle.distributed.launch --gpus ${gpu_list} \
        --log_dir ${distributed_log_dir} \
        ${train_script} \
        --device ${device} \
        --save_dir ${save_dir} \
        --batch_size ${batch_size} \
        --epoch ${epoch}
    echo -e "\nend ernie text classfication training."
elif [ ${process_name} == ernie_text_cls_export ]; then
    echo -e "\nstart to export model for ernie text classfication."
    python ${export_script} --params_path=${params_path} --output_path=${static_output_path}
    echo -e "\nend to export model for ernie text classfication."
elif [ ${process_name} == ernie_text_cls_predict ]; then
    echo -e "\nstart to ernie text classfication prediction."
    python ${predict_script} --device ${device} \
        --model_file=${static_model_file} \
        --params_file=${static_params_file}
    echo -e "\nend to ernie text classfication prediction."
else
    echo "no process name ${process_name} ."
fi