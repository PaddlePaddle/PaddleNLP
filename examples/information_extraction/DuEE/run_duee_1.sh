dataset_name=DuEE1.0
data_dir=./data/${dataset_name}
conf_dir=./conf/${dataset_name}
ckpt_dir=./ckpt/${dataset_name}
submit_data_path=./submit/test_duee_1.json
pred_data=${data_dir}/duee_test1.json   # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=16
epoch=20

echo -e "check and create directory"
dir_list=(./ckpt ${ckpt_dir} ./submit)
for item in ${dir_list[*]}
do
    if [ ! -d ${item} ]; then
        mkdir ${item}
        echo "create dir * ${item} *"
    else
        echo "dir ${item} exist"
    fi
done

process_name=${1}

run_sequence_labeling_model(){
    model=${1}
    is_train=${2}
    pred_save_path=${ckpt_dir}/${model}/test_pred.json
    sh run_sequence_labeling.sh ${data_dir}/${model} ${conf_dir}/${model}_tag.dict ${ckpt_dir}/${model} ${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}
}

if [ ${process_name} == data_prepare ]; then
    echo -e "\nstart ${dataset_name} data prepare"
    python duee_1_data_prepare.py
    echo -e "end ${dataset_name} data prepare"
elif [ ${process_name} == trigger_train ]; then
    echo -e "\nstart ${dataset_name} trigger train"
    run_sequence_labeling_model trigger True
    echo -e "end ${dataset_name} trigger train"
elif [ ${process_name} == trigger_predict ]; then
    echo -e "\nstart ${dataset_name} trigger predict"
    run_sequence_labeling_model trigger False
    echo -e "end ${dataset_name} trigger predict"
elif [ ${process_name} == role_train ]; then
    echo -e "\nstart ${dataset_name} role train"
    run_sequence_labeling_model role True
    echo -e "end ${dataset_name} role train"
elif [ ${process_name} == role_predict ]; then
    echo -e "\nstart ${dataset_name} role predict"
    run_sequence_labeling_model role False
    echo -e "end ${dataset_name} role predict"
elif [ ${process_name} == pred_2_submit ]; then
    echo -e "\nstart ${dataset_name} predict data merge to submit fotmat"
    python duee_1_postprocess.py --trigger_file ${ckpt_dir}/trigger/test_pred.json --role_file ${ckpt_dir}/role/test_pred.json --schema_file ${conf_dir}/event_schema.json --save_path ${submit_data_path}
    echo -e "end ${dataset_name} role predict data merge"
else
    echo "no process name ${process_name}"
fi