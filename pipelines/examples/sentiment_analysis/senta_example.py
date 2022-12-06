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

import argparse
import logging
import os
import sys
sys.path.insert(1, "./../..")
sys.path.insert(2, "./../../..")

import paddle
from pipelines.nodes import SentaProcessor, UIESenta
from pipelines import SentaPipeline

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run docprompt system, defaults to gpu.")
parser.add_argument("--batch_size", default=4, type=int, help="The batch size of prompt for one image.")
args = parser.parse_args()
# yapf: enable


def senta_pipeline():

    preprocessor = SentaProcessor(max_examples=10)
    schema = [{'评价维度': ['观点词', '情感倾向[正向,负向,未提及]']}]
    senta = UIESenta(schema=schema, model="uie-base")

    pipe = SentaPipeline(preprocessor=preprocessor, senta=senta)
    
    # image link input
    text = "蛋糕味道不错，店家服务也很好"
    meta = {
        "file_path": "/wangqinghui/mynlp/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/data/test_hotel.txt",
        "save_path": ""
    }
    # image local path input
    # meta = {"doc": "./invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]}

    prediction = pipe.run(meta=meta)
    print(prediction)
    print(prediction["results"][0])


if __name__ == "__main__":
    senta_pipeline()
