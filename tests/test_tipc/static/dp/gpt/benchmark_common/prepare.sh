cd ../examples/language_model/gpt/data_tools/
sed -i "s/python3/python/g" Makefile
sed -i "s/python-config/python3.7m-config/g" Makefile
cd -

cd ../examples/language_model/gpt-3/data_tools/
sed -i "s/python3/python/g" Makefile
sed -i "s/python-config/python3.7m-config/g" Makefile
cd -

mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy -o .tmp
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz -o .tmp
cd -

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install setuptools_scm 
python -m pip install Cython 
python -m pip install -r ../requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pybind11 regex sentencepiece tqdm visualdl attrdict pyyaml -i https://mirror.baidu.com/pypi/simple

python -m pip install -e ../
# python -m pip install paddlenlp    # PDC 镜像中安装失败
python -m pip list
