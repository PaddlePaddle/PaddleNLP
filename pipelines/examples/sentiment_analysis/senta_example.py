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

from pipelines import SentaPipeline
from pipelines.nodes import SentaProcessor, SentaVisualization, UIESenta


def format_print(results):
    """
    Print Information in results.
    """
    if "sr_save_path" in results:
        print("\nText Result: ", results["sr_save_path"])
    if "img_dict" in results:
        print("Visualization Result: ")
        for img_name in results["img_dict"]:
            print("\t{}:{}".format(img_name, results["img_dict"][img_name]))


def senta_pipeline(args):
    """
    Sentiment Analysis with Pipeline.
    """

    # initializing SentaPipeline
    preprocessor = SentaProcessor(max_examples=args.max_examples)
    if not args.aspects:
        schema = [{"评价维度": ["观点词", "情感倾向[正向,负向,未提及]"]}]
        senta = UIESenta(schema=schema, model=args.model, batch_size=args.batch_size, aspects=args.aspects)
    else:
        schema = ["观点词", "情感倾向[正向,负向,未提及]"]
        senta = UIESenta(schema=schema, model=args.model, batch_size=args.batch_size)
    visualization = SentaVisualization(font_name="SimHei")
    senta_pipeline = SentaPipeline(preprocessor=preprocessor, senta=senta, visualization=visualization)

    # run SentaPipeline for inputting file.
    meta = {"file_path": args.file_path}
    results = senta_pipeline.run(meta=meta)
    format_print(results)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True, type=str, help="The file that you want to perform sentiment analysis on.")
    parser.add_argument("--max_examples", default=-1, type=int, help="The maxinum number of examples processed by pipline.")
    parser.add_argument("--model", choices=['uie-senta-base', 'uie-senta-medium', 'uie-senta-mini', 'uie-senta-micro', 'uie-senta-nano'], default="uie-senta-base", help="The model name that you wanna use for sentiment analysis.")
    parser.add_argument("--aspects", default=None, type=str, nargs="+", help="A list of pre-given aspects, that is to say, Pipeline only perform sentiment analysis on these pre-given aspects if you input it.")
    parser.add_argument("--batch_size", default=4, type=int, help="Number of samples the model receives in one batch for sentiment inference.")

    args = parser.parse_args()
    # yapf: enable

    senta_pipeline(args)
