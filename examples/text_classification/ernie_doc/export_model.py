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
import os
import paddle
import shutil
from paddlenlp.utils.log import logger
from predict import LongDocClassifier

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for predicting (In static mode, it should be the same as in model training process.)")
parser.add_argument("--model_name_or_path", type=str, default="ernie-doc-base-zh",
                    help="Pretraining or finetuned model name or path")
parser.add_argument("--max_seq_length", type=int, default=512,
                    help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--memory_length", type=int, default=128, help="Length of the retained previous heads.")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"],
                    help="Select cpu, gpu devices to train model.")
parser.add_argument("--dataset", default="iflytek", choices=["imdb", "iflytek", "thucnews", "hyp"], type=str,
                    help="The training dataset")
parser.add_argument("--static_path", default=None, type=str,
                    help="The path which your static model is at or where you want to save after converting.")

args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    paddle.set_device(args.device)

    if os.path.exists(args.model_name_or_path):
        logger.info("init checkpoint from %s" % args.model_name_or_path)

    if args.static_path and os.path.exists(args.static_path):
        logger.info("will remove the old model")
        shutil.rmtree(args.static_path)

    predictor = LongDocClassifier(model_name_or_path=args.model_name_or_path,
                                  batch_size=args.batch_size,
                                  max_seq_length=args.max_seq_length,
                                  memory_len=args.memory_length,
                                  static_mode=True,
                                  static_path=args.static_path)
