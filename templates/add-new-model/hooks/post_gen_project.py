# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil

tokenier_type = "{{cookiecutter.tokenizer_type}}"
test_tokenizer_files = {
    "Based on Bert": "test_bert_tokenizer.py",
    "Based on BPETokenizer": "test_bpe_tokenizer.py",
    "Based on SentencePiece": "test_sp_tokenizer.py",
}

assert tokenier_type in test_tokenizer_files

# copy the file
shutil.copyfile(
    os.path.join("test_tokenizers", test_tokenizer_files[tokenier_type]),
    "test_tokenizer.py"
)

shutil.rmtree("test_tokenizers", ignore_errors=True)
