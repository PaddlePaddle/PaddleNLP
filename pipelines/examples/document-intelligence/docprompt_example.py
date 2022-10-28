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

import paddle
from pipelines.nodes import DocOCRProcessor, DocPrompter
from pipelines import DocPipeline

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run docprompt system, defaults to gpu.")
parser.add_argument("--batch_size", default=4, type=int, help="The batch size of prompt for one image.")
args = parser.parse_args()
# yapf: enable


def docprompt_pipeline():

    use_gpu = True if args.device == 'gpu' else False

    preprocessor = DocOCRProcessor(use_gpu=use_gpu)
    docprompter = DocPrompter(use_gpu=use_gpu, batch_size=args.batch_size)
    pipe = DocPipeline(preprocessor=preprocessor, modelrunner=docprompter)
    # image link input
    meta = {
        "doc":
        "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/images/invoice.jpg",
        "prompt": ["发票号码是多少?", "校验码是多少?"]
    }
    # image local path input
    # meta = {"doc": "./invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]}

    prediction = pipe.run(meta=meta)
    print(prediction["results"][0])


if __name__ == "__main__":
    docprompt_pipeline()
