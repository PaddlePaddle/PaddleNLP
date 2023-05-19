# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import sys

from utils import rouge_eval, rouge_log

decode_dir = sys.argv[1]

print("Decoder has finished reading dataset for single_pass.")
print("Now starting ROUGE eval...")
results_dict = rouge_eval(decode_dir + "rouge_ref", decode_dir + "rouge_dec_dir")
rouge_log(results_dict, decode_dir + "rouge_dec_dir")
