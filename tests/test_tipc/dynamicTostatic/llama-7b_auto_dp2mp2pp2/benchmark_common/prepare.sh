# install可选
cd ../  # PaddleNLP 根目录
pwd
pip install -e .
cd -

# 下载、解压、拷贝必要数据集
cd ../llm/llama/auto_parallel/
# llama 模型数据下载
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz

mkdir data
mv llama_openwebtext_100k_ids.npy ./data
mv llama_openwebtext_100k_idx.npz ./data
cd -