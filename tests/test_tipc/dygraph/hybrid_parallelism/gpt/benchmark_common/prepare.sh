cd ../examples/language_model/gpt-3/data_tools/
sed -i "s/python3/python3.7/g" Makefile
cd -

python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
unset http_proxy https_proxy
python3 -m pip install -r ../requirements.txt #-i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install pybind11 regex sentencepiece tqdm visualdl #-i https://mirror.baidu.com/pypi/simple
python3 -m pip install --upgrade paddlenlp
# get data
cd ../examples/language_model/gpt-3/dygraph/
rm -rf data
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/train.data.json_ids.npz
