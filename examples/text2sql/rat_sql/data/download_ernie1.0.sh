#!/usr/bin/env bash

cd `dirname $0`

model_files_path="ernie/ernie_1.0_base_ch"

#get pretrained ernie1.0 model params
wget --no-check-certificate https://ernie-github.cdn.bcebos.com/model-ernie1.0.1.tar.gz
if [ ! -d $model_files_path ]; then
	mkdir -p $model_files_path
fi
tar xzf model-ernie1.0.1.tar.gz -C $model_files_path
rm model-ernie1.0.1.tar.gz

