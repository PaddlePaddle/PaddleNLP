#!/bin/bash

# download baseline model file to ./model_baseline/
if [ -d ./model_baseline/ ]
then
    echo "./model_baseline/ directory already existed, ignore download"
else
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis-2.0.0.tar.gz
    tar xvf lexical_analysis-2.0.0.tar.gz
    /bin/rm lexical_analysis-2.0.0.tar.gz
fi

# download dataset file to ./data/
if [ -d ./data/ ]
then
    echo "./data/ directory already existed, ignore download"
else
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis-dataset-2.0.0.tar.gz
    tar xvf lexical_analysis-dataset-2.0.0.tar.gz
    /bin/rm lexical_analysis-dataset-2.0.0.tar.gz
fi

# download ERNIE pretrained model to ./pretrained/
if [ -d ./pretrained/ ]
then
    echo "./pretrained/ directory already existed, ignore download"
else
    mkdir ./pretrained/ && cd ./pretrained/
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz
    tar xvf ERNIE_stable-1.0.1.tar.gz
    /bin/rm ERNIE_stable-1.0.1.tar.gz
    cd ../
fi

# download finetuned model file to ./model_finetuned/
if [ -d ./model_finetuned/ ]
then
    echo "./model_finetuned/ directory already existed, ignored download"
else
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis_finetuned-1.0.0.tar.gz
    tar xvf lexical_analysis_finetuned-1.0.0.tar.gz
    /bin/rm lexical_analysis_finetuned-1.0.0.tar.gz
fi
