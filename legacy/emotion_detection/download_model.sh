#!/bin/bash

mkdir -p pretrain_models
cd pretrain_models

# download pretrain model file to ./models/
MODEL_CNN=https://baidu-nlp.bj.bcebos.com/emotion_detection_textcnn-1.0.0.tar.gz
MODEL_ERNIE=https://baidu-nlp.bj.bcebos.com/emotion_detection_ernie_finetune-1.0.0.tar.gz
wget --no-check-certificate ${MODEL_CNN}
wget --no-check-certificate ${MODEL_ERNIE}

tar xvf emotion_detection_textcnn-1.0.0.tar.gz
tar xvf emotion_detection_ernie_finetune-1.0.0.tar.gz

/bin/rm emotion_detection_textcnn-1.0.0.tar.gz
/bin/rm emotion_detection_ernie_finetune-1.0.0.tar.gz
