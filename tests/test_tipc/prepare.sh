#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer']
# PaddleNLP supports 'lite_train_lite_infer', 'lite_train_whole_infer', 'whole_train_whole_infer' and 
# 'whole_infer' mode now.

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

if [ ${MODE} = "lite_train_lite_infer" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
        cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd .. 
    fi

    if [[ ${model_name} =~ transformer* ]]; then
        cd ../examples/machine_translation/transformer/

        # The whole procedure of lite_train_infer should be less than 15min.
        # Hence, set maximum output length is 16. 
        sed -i "s/^max_out_len.*/max_out_len: 16/g" configs/transformer.base.yaml
        sed -i "s/^max_out_len.*/max_out_len: 16/g" configs/transformer.big.yaml

        sed -i "s/^random_seed:.*/random_seed: 128/g" configs/transformer.base.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: False/g" configs/transformer.base.yaml
        sed -i "s/^shuffle:.*/shuffle: False/g" configs/transformer.base.yaml

        sed -i "s/^random_seed:.*/random_seed: 128/g" configs/transformer.big.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: False/g" configs/transformer.big.yaml
        sed -i "s/^shuffle:.*/shuffle: False/g" configs/transformer.big.yaml

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
elif [ ${MODE} = "whole_train_whole_infer" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
        cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd ..
    fi

    if [[ ${model_name} =~ transformer* ]]; then
        cd ../examples/machine_translation/transformer/
        sed -i "s/^max_out_len.*/max_out_len: 256/g" configs/transformer.base.yaml
        sed -i "s/^max_out_len.*/max_out_len: 1024/g" configs/transformer.big.yaml

        sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.base.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.base.yaml
        sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.base.yaml

        sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.big.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.big.yaml
        sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.big.yaml

        # Whole data set prepared. 
        if [ ! -f WMT14.en-de.tar.gz ]; then
            wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.tar.gz
            tar -zxf WMT14.en-de.tar.gz
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
        cp -f WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33712 ./
        cp -f WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 ./
        # Train with whole data. 
        ln -s WMT14.en-de/wmt14_ende_data_bpe/train.tok.clean.bpe.33708.en train.en
        ln -s WMT14.en-de/wmt14_ende_data_bpe/train.tok.clean.bpe.33708.de train.de
        # Dev with whole data. 
        ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2013.tok.bpe.33708.en dev.en
        ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2013.tok.bpe.33708.de dev.de
        # Test with whole data. 
        ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
        ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
        cd -
    fi
elif [ ${MODE} = "lite_train_whole_infer" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
        cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd ..
    fi

    if [[ ${model_name} =~ transformer* ]]; then
        cd ../examples/machine_translation/transformer/
        sed -i "s/^max_out_len.*/max_out_len: 256/g" configs/transformer.base.yaml
        sed -i "s/^max_out_len.*/max_out_len: 1024/g" configs/transformer.big.yaml

        sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.base.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.base.yaml
        sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.base.yaml

        sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.big.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.big.yaml
        sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.big.yaml

        # Trained transformer base model checkpoint. 
        # For infer. 
        if [ ! -f transformer-base-wmt_ende_bpe.tar.gz ]; then
            wget https://bj.bcebos.com/paddlenlp/models/transformers/transformer/transformer-base-wmt_ende_bpe.tar.gz
            tar -zxf transformer-base-wmt_ende_bpe.tar.gz
            mv base_trained_models/ trained_models/
        fi
        # For train. 
        if [ ! -f WMT14.en-de.partial.tar.gz ]; then
            wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.partial.tar.gz
            tar -zxf WMT14.en-de.partial.tar.gz
        fi
        # Whole data set prepared. 
        if [ ! -f WMT14.en-de.tar.gz ]; then
            wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.tar.gz
            tar -zxf WMT14.en-de.tar.gz
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
        # Train with partial data. 
        ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.en train.en
        ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.de train.de
        # Dev with partial data. 
        ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.en dev.en
        ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.de dev.de
        # Test with whole data. 
        ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
        ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
        cd -
    fi
elif [ ${MODE} = "whole_infer" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
        cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd ..
        # Download static model
        rm -rf ./test_tipc/bigru_crf/infer_model
        wget -nc -P ./test_tipc/bigru_crf/ https://bj.bcebos.com/paddlenlp/models/bigru_crf_infer_model.tgz  --no-check-certificate
        cd ./test_tipc/bigru_crf && tar xfz bigru_crf_infer_model.tgz && cd ../..
    fi

    if [[ ${model_name} =~ transformer* ]]; then
        cd ../examples/machine_translation/transformer/
        sed -i "s/^max_out_len.*/max_out_len: 256/g" configs/transformer.base.yaml
        sed -i "s/^max_out_len.*/max_out_len: 1024/g" configs/transformer.big.yaml

        sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.base.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.base.yaml
        sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.base.yaml

        sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.big.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.big.yaml
        sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.big.yaml

        # Trained transformer base model checkpoint. 
        if [ ! -f transformer-base-wmt_ende_bpe.tar.gz ]; then
            wget https://bj.bcebos.com/paddlenlp/models/transformers/transformer/transformer-base-wmt_ende_bpe.tar.gz
            tar -zxf transformer-base-wmt_ende_bpe.tar.gz
            mv base_trained_models/ trained_models/
        fi
        # Whole data set prepared. 
        if [ ! -f WMT14.en-de.tar.gz ]; then
            wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.tar.gz
            tar -zxf WMT14.en-de.tar.gz
        fi
        # Set soft link.
        if [ -f test.en ]; then
            rm -f test.en
        fi
        if [ -f test.de ]; then
            rm -f test.de
        fi
        rm -f vocab_all.bpe.33712
        rm -f vocab_all.bpe.33708
        # Vocab
        cp -f WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33712 ./
        cp -f WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 ./
        # Test with whole data. 
        ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
        ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
        cd -
    fi
elif [ ${MODE} = "benchmark_train" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py --remote-path frame_benchmark/paddle/PaddleNLP/lexical_analysis_dataset_tiny/ --local-path ./data/ --mode download
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

        python -m pip install h5py -i https://mirror.baidu.com/pypi/simple
    fi

    if [[ ${model_name} == "gpt2" ]]; then
        cd ../examples/language_model/gpt/data_tools/
        sed -i "s/python3/python/g" Makefile
        sed -i "s/python-config/python3.7m-config/g" Makefile
        cd -
        mkdir -p data && cd data
        wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy -o .tmp
        wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz -o .tmp
        cd -
    fi

    if [[ ${model_name} == "gpt3" ]]; then
        cd ../examples/language_model/gpt-3/data_tools/
        sed -i "s/python3/python/g" Makefile
        sed -i "s/python-config/python3.7m-config/g" Makefile
        cd -
        mkdir -p data && cd data
        wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy -o .tmp
        wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz -o .tmp
        cd -
    fi

    if [[ ${model_name} =~ transformer* ]]; then
        cd ../examples/machine_translation/transformer/

        git checkout .

        sed -i "s/^random_seed:.*/random_seed: 128/g" configs/transformer.base.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: False/g" configs/transformer.base.yaml
        sed -i "s/^shuffle:.*/shuffle: False/g" configs/transformer.base.yaml

        sed -i "s/^random_seed:.*/random_seed: 128/g" configs/transformer.big.yaml
        sed -i "s/^shuffle_batch:.*/shuffle_batch: False/g" configs/transformer.big.yaml
        sed -i "s/^shuffle:.*/shuffle: False/g" configs/transformer.big.yaml

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
    python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install setuptools_scm 
    python -m pip install Cython 
    python -m pip install -r ../requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install pybind11 regex sentencepiece tqdm visualdl attrdict pyyaml -i https://mirror.baidu.com/pypi/simple

    python -m pip install -e ../
    # python -m pip install paddlenlp    # PDC 镜像中安装失败
    python -m pip list
fi
