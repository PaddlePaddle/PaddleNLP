# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eux

rm 1-billion-word* -rf
wget https://bj.bcebos.com/paddlenlp/datasets/1-billion-word.tar.gz
src_md5="5f079a9b88ea27585e0539f502ca9327"
md5=`md5sum 1-billion-word.tar.gz | cut -d ' ' -f1`
if [ $md5 != $src_md5 ]
then
    echo "The MD5 values of 1-billion-word.tar.gz are inconsistent. Please download again!"
    exit 1
fi
tar -zxf 1-billion-word.tar.gz

rm sentence-polarity-dataset-v1* -rf
wget https://bj.bcebos.com/paddlenlp/datasets/movie-review/sentence-polarity-dataset-v1.tar.gz
src_md5="0464239d7b14b18d941f54a948c6cb26"
md5=`md5sum sentence-polarity-dataset-v1.tar.gz | cut -d ' ' -f1`
if [ $md5 != $src_md5 ]
then
    echo "The MD5 values of sentence-polarity-dataset-v1.tar.gz are inconsistent. Please download again!"
    exit 1
fi
tar -zxf sentence-polarity-dataset-v1.tar.gz
cd sentence-polarity-dataset-v1
gunzip GoogleNews-vectors-negative300.bin.gz
