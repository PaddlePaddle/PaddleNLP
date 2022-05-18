# prepare
python3 -m pip install -r ../../../../../requirements.txt
python3 -m pip install pybind11
python3 -m pip install regex sentencepiece tqdm visualdl

# get data
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/train.data.json_ids.npz
cd ..

