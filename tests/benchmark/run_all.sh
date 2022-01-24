# Test training benchmark for several models.


# Usage:
#   git clone https://github.com/PaddlePaddle/PaddleNLP.git
#   cd PaddleNLP
#   bash tests/benchmark/run_all.sh static|dygraph


################################# 配置python, 如:
#rm -rf $run_env
#mkdir -p $run_env
#echo `which python3.7`
#ln -s $(which python3.7)m-config  $run_env/python3-config
##ln -s /usr/local/python3.7.0/lib/python3.7m-config /usr/local/bin/python3-config
#ln -s $(which python3.7) $run_env/python
#ln -s $(which pip3.7) $run_env/pip
#export PATH=$run_env:${PATH}

pip install -r requirements.txt
pip install pybind11 regex sentencepiece tqdm visualdl
pip install -e ./

# Download test dataset and save it to PaddleNLP/data
if [ -d data ]; then
    rm -rf data
fi
mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy -o .tmp
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz -o .tmp
cd -

mode_item=$1   #static|dygraph

if [ ${mode_item} != "static" ] && [ ${mode_item} != "dygraph" ]; then
    echo "please set mode_item(static|dygraph)"
    exit 1
fi
profile="off"
model_list=(gpt2 gpt3)
fp_list=(fp16)

for model_item in ${model_list[@]}
do
    for fp_item in ${fp_list[@]}
    do
	    if [ ${mode_item} == "static" ] && [ ${fp_item} == "fp16" ]; then
	        bs_item=16
            else
                bs_item=8
            fi
            CUDA_VISIBLE_DEVICES=0 bash tests/benchmark/run_benchmark.sh sp ${bs_item} ${fp_item}  200  ${model_item} ${mode_item} ${profile} | tee ${log_path}/${model_item}_${mode_item}_bs${bs_item}_${fp_item}_1gpus 2>&1
            sleep 10
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tests/benchmark/run_benchmark.sh mp ${bs_item} ${fp_item}  200  ${model_item} ${mode_item} ${profile} | tee ${log_path}/${model_item}_${mode_item}_bs${bs_item}_${fp_item}_8gpus8p 2>&1
done
done
