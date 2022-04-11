#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

if [ ${model_name} == "bigru_crf" ]; then
    rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
    wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
    cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd .. 
fi

if [[ ${model_name} =~ bert* ]]; then
    rm -rf ./data/wikicorpus_en_seqlen128/ wikicorpus_en_seqlen128.tar wikicorpus_en_seqlen512 hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/ hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar
    wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/benchmark_wikicorpus_en_seqlen128.tar --no-check-certificate
    wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/benchmark_hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar --no-check-certificate

    cd ./data/
    tar -xf benchmark_wikicorpus_en_seqlen128.tar
    tar -xf benchmark_hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar

    ln -s hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en_seqlen512/ wikicorpus_en_seqlen512

    cd ..
fi

if [[ ${model_name} =~ gpt* ]]; then
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
fi

if [[ ${model_name} =~ transformer* ]]; then
    cd ../examples/machine_translation/transformer/

    # Data set prepared. 
    if [ ! -f WMT14.en-de.partial.tar.gz ]; then
        wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.partial.tar.gz
        tar -zxf WMT14.en-de.partial.tar.gz
    fi
    # Set soft link.
    if [ -f train.en ]; then
        rm -f train.en
    fi
    if [ -f train.de ]; then
        rm -f train.de
    fi
    if [ -f dev.en ]; then
        rm -f dev.en
    fi
    if [ -f dev.de ]; then
        rm -f dev.de
    fi
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    rm -f vocab_all.bpe.33712
    rm -f vocab_all.bpe.33708
    # Vocab
    cp -f WMT14.en-de.partial/wmt14_ende_data_bpe/vocab_all.bpe.33712 ./
    cp -f WMT14.en-de.partial/wmt14_ende_data_bpe/vocab_all.bpe.33708 ./
    # Train
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.en train.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.de train.de
    # Dev
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.en dev.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.de dev.de
    #Test
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/test.tok.bpe.en test.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/test.tok.bpe.de test.de
    cd -
fi

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python3.7 -m pip install --upgrade pip
python3.7 -m pip install -r ../requirements.txt -i https://mirror.baidu.com/pypi/simple
python3.7 -m pip install pybind11 regex sentencepiece tqdm visualdl attrdict pyyaml -i https://mirror.baidu.com/pypi/simple
python3.7 -m pip install -e ..
