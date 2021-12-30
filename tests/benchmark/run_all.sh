# Test training benchmark for several models.

# Use docker： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37

# Usage:
#   git clone https://github.com/PaddlePaddle/PaddleNLP.git
#   cd PaddleNLP
#   bash tests/benchmark/run_all.sh


profile=${1:-"off"}

export BENCHMARK_ROOT=/workspace
run_env=$BENCHMARK_ROOT/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=paddle2.1.3
cuda_version=10.2
save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/

if [[ -d ${save_log_dir} ]]; then
    rm -rf ${save_log_dir}
fi
# this for update the log_path coding mat
export TRAIN_LOG_DIR=${save_log_dir}/train_log
mkdir -p ${TRAIN_LOG_DIR}
log_path=${TRAIN_LOG_DIR}

################################# 配置python, 如:
rm -rf $run_env
mkdir $run_env
echo `which python3.7`
ln -s $(which python3.7)m-config  $run_env/python3-config
#ln -s /usr/local/python3.7.0/lib/python3.7m-config /usr/local/bin/python3-config
ln -s $(which python3.7) $run_env/python
ln -s $(which pip3.7) $run_env/pip

export PATH=$run_env:${PATH}

#pip install -r requirements.txt
cd $BENCHMARK_ROOT
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
pip install pybind11 regex sentencepiece tqdm visualdl -i https://mirror.baidu.com/pypi/simple
pip install -e ./

# Download test dataset and save it to PaddleNLP/data
if [ -d data ]; then
    rm -rf data
fi
mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy -o .tmp
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz -o .tmp
cd -

model_name='nlp'
mode_list=(static dygraph)
repo_list=(gpt2 gpt3) # gpt3 is optimized for speed and need paddle develop version
max_iters=200 # control the test time

SP_CARDNUM='0'
MP_CARDNUM='0,1,2,3,4,5,6,7'
for mod_item in ${mode_list[@]}; do
    CUDA_VISIBLE_DEVICES=$SP_CARDNUM bash tests/benchmark/run_benchmark.sh sp 8 fp32  ${max_iters} ${model_name} ${mod_item} ${profile}
    CUDA_VISIBLE_DEVICES=$MP_CARDNUM bash tests/benchmark/run_benchmark.sh mp 8 fp32 ${max_iters} ${model_name} ${mod_item} ${profile} 
    if [ $mod_item == 'dygraph' ]; then
        # now, in dygraph mod, the bs=16 will out of mem in 32G V100
        CUDA_VISIBLE_DEVICES=$SP_CARDNUM bash tests/benchmark/run_benchmark.sh sp 8 fp16  ${max_iters} ${model_name} ${mod_item} ${profile}
        CUDA_VISIBLE_DEVICES=$MP_CARDNUM bash tests/benchmark/run_benchmark.sh mp 8 fp16 ${max_iters} ${model_name} ${mod_item} ${profile}
    else
        CUDA_VISIBLE_DEVICES=$SP_CARDNUM bash tests/benchmark/run_benchmark.sh sp 16 fp16  ${max_iters} ${model_name} ${mod_item} ${profile}
        CUDA_VISIBLE_DEVICES=$MP_CARDNUM bash tests/benchmark/run_benchmark.sh mp 16 fp16 ${max_iters} ${model_name} ${mod_item} ${profile}
    fi
done

# gpt-3 need the latest paddlepaddle develop
wget https://paddle-wheel.bj.bcebos.com/develop/linux/gpu-cuda10.2-cudnn7-mkl_gcc8.2/paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-linux_x86_64.whl -o .tmp
python3 -m pip install paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-linux_x86_64.whl  --upgrade
rm paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-linux_x86_64.whl 

for mod_item in ${mode_list[@]}; do
    CUDA_VISIBLE_DEVICES=$SP_CARDNUM bash tests/benchmark/run_benchmark.sh sp 8 fp32  ${max_iters} ${model_name} ${mod_item} ${profile} gpt3
    CUDA_VISIBLE_DEVICES=$MP_CARDNUM bash tests/benchmark/run_benchmark.sh mp 8 fp32 ${max_iters} ${model_name} ${mod_item} ${profile} gpt3
    if [ $mod_item == 'dygraph' ]; then
        # now, in dygraph mod, the bs=16 will out of mem in 32G V100
        CUDA_VISIBLE_DEVICES=$SP_CARDNUM bash tests/benchmark/run_benchmark.sh sp 8 fp16  ${max_iters} ${model_name} ${mod_item} ${profile} gpt3
        CUDA_VISIBLE_DEVICES=$MP_CARDNUM bash tests/benchmark/run_benchmark.sh mp 8 fp16 ${max_iters} ${model_name} ${mod_item} ${profile} gpt3
    else
        CUDA_VISIBLE_DEVICES=$SP_CARDNUM bash tests/benchmark/run_benchmark.sh sp 16 fp16  ${max_iters} ${model_name} ${mod_item} ${profile} gpt3
        CUDA_VISIBLE_DEVICES=$MP_CARDNUM bash tests/benchmark/run_benchmark.sh mp 16 fp16 ${max_iters} ${model_name} ${mod_item} ${profile} gpt3
    fi
done
