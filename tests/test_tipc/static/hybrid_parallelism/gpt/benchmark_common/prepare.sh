cd ../examples/language_model/gpt-3/data_tools/
sed -i "s/python/python3.7/g" Makefile
sed -i "s/python-config/python3.7-config/g" Makefile
cd -

python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install -r ../requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install pybind11 regex sentencepiece tqdm visualdl -i https://mirror.baidu.com/pypi/simple

# get data
cd ../examples/language_model/gpt-3/static/
mkdir train_data && cd train_data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/train.data.json_ids.npz
