export PYTHONPATH=/mnt/zhangxuefei/program-paddle/PaddleNLP/
export CUDA_VISIBLE_DEVICES=1


echo -e "check and create directory"
if [ ! -d ./test_log_4 ]; then
    mkdir test_log_4
    echo "create dir * ./test_log_4 *"
else
    echo "dir ./test_log_4 exist"
fi

seq_lens=(10 32 64 128 256 512)
batch_sizes=(1 4 8 16 32 64 128)
for seq_len in ${seq_lens[*]}; do
python experimental/export_model.py --max_seq_length ${seq_len} \
    --params_path experimental/checkpoint/model_900.pdparams \
    --output_path experimental/export
python text_classification/pretrained_models/export_model.py \
    --params_path text_classification/pretrained_models/checkpoint/model_900/model_state.pdparams \
    --output_path text_classification/pretrained_models/export
    for batch_size in ${batch_sizes[*]}; do

        echo "=========================================================="
        echo "do fast predict on seq_len:${seq_len} , batch_size:${batch_size}"
        echo "=========================================================="

        save_log_path="./test_log_4/fast_seq_len_${seq_len}_batch_size_${batch_size}.log"
python experimental/python_deploy.py --model_dir experimental/export \
    --batch_size ${batch_size} --max_seq_length ${seq_len} > ${save_log_path} 2>&1

        echo "=========================================================="
        echo "do predict on seq_len:${seq_len} , batch_size:${batch_size}"
        echo "=========================================================="

        save_log_path="./test_log_4/py_seq_len_${seq_len}_batch_size_${batch_size}.log"
        python text_classification/pretrained_models/deploy/python/predict.py --model_dir text_classification/pretrained_models/export \
            --batch_size ${batch_size} --max_seq_length ${seq_len} > ${save_log_path} 2>&1
    done
done
