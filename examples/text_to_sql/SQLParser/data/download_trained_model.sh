#!/bin/bash

cd `dirname $0`

version='v1.0.0'
target_file="text2sql_trained_model_$version.tar.gz"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/text2sql/$target_file
tar xzf $target_file
rm -rf $target_file

