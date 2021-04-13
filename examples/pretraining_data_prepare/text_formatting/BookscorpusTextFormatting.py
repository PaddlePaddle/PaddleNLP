# Copyright (c) 2019 NVIDIA CORPORATION & 2021 PaddlePaddle Authors. 
# All rights reserved.
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

import glob
import os

from datasets import load_dataset


class BookscorpusTextFormatting:
    def __init__(self, output_filename):
        self.bookcorpus_ds = load_dataset("bookcorpus")
        self.books_path = books_path
        self.output_filename = output_filename

    # This puts one book per line
    def merge(self):
        with open(self.output_filename, mode='w', newline='\n') as ofile:
            for data in self.bookcorpus_ds['train']:
                text = data['text']
                if text.strip() != "":
                    ofile.write(line.strip() + ' \n')
