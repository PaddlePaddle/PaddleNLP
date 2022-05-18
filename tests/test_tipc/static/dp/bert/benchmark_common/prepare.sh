rm -rf ./data/wikicorpus_en_seqlen128/ wikicorpus_en_seqlen128.tar wikicorpus_en_seqlen512 hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/ hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar
wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/benchmark_wikicorpus_en_seqlen128.tar --no-check-certificate
wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/benchmark_hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar --no-check-certificate

cd ./data/
tar -xf benchmark_wikicorpus_en_seqlen128.tar
tar -xf benchmark_hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar

ln -s hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en_seqlen512/ wikicorpus_en_seqlen512

cd ..

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install setuptools_scm 
python -m pip install Cython 
python -m pip install -r ../requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pybind11 regex sentencepiece tqdm visualdl attrdict pyyaml h5py -i https://mirror.baidu.com/pypi/simple

python -m pip install -e ../
# python -m pip install paddlenlp    # PDC 镜像中安装失败
python -m pip list
