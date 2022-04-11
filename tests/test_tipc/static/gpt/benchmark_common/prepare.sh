export BENCHMARK_ROOT=/workspace
mkdir -p $BENCHMARK_ROOT
run_env=$BENCHMARK_ROOT/run_env

rm -rf $run_env
mkdir $run_env

echo `which python3.7`

ln -s $(which python3.7)m-config  $run_env/python3-config
# ln -s /usr/local/python3.7.0/lib/python3.7m-config /usr/local/bin/python3-config

ln -s $(which python3.7) $run_env/python
ln -s $(which python3.7) $run_env/python3
ln -s $(which pip3.7) $run_env/pip

export PATH=$run_env:${PATH}

mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy -o .tmp
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz -o .tmp
cd -

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python -m pip install --upgrade pip
python -m pip install -r ../requirements.txt -i https://mirror.baidu.com/pypi/simple
python -m pip install pybind11 regex sentencepiece tqdm visualdl attrdict pyyaml -i https://mirror.baidu.com/pypi/simple
python -m pip install -e ..
