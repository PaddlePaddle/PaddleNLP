#!/bin/bash

# download dataset file to ./data/
DATA_URL=https://baidu-nlp.bj.bcebos.com/emotion_detection-dataset-1.0.0.tar.gz
wget --no-check-certificate ${DATA_URL}

tar xvf emotion_detection-dataset-1.0.0.tar.gz
/bin/rm emotion_detection-dataset-1.0.0.tar.gz
