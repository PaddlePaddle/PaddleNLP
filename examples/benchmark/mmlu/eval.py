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

# Adapted from https://github.com/ymcui/Chinese-LLaMA-Alpaca and https://github.com/SJTU-LIT/ceval
import argparse
import json
import os

import numpy as np
import pandas as pd
from categories import categories, subcategories
from evaluator import ModelEvaluator

choices = ["A", "B", "C", "D"]


def main(args, evaluator):
    subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f]
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "results_{}".format(args.model_name_or_path))):
        os.makedirs(os.path.join(args.output_dir, "results_{}".format(args.model_name_or_path)))

    all_cors = []
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}
    summary = {}
    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[: args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = evaluator.eval(args, subject, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model_name_or_path)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model_name_or_path, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(args.output_dir, "results_{}".format(args.model_name_or_path), "{}.csv".format(subject)),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
        summary[subcat] = {
            "acc:": subcat_acc,
            "correct:": np.sum(np.concatenate(subcat_cors[subcat])),
            "num:": np.concatenate(subcat_cors[subcat]).size,
        }

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    print("Model:", args.model_name_or_path)
    summary["All"] = {
        "acc:": weighted_acc,
        "correct:": np.sum(np.concatenate(all_cors)),
        "num:": np.concatenate(all_cors).size,
    }
    json.dump(
        summary,
        open(os.path.join(args.output_dir, "results_{}".format(args.model_name_or_path), "summary.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--dtype", default="float32", type=str)

    args = parser.parse_args()
    print(args)

    evaluator = ModelEvaluator(
        model_name_or_path=args.model_name_or_path,
        ntrain=args.ntrain,
        temperature=args.temperature,
        dtype=args.dtype,
    )

    main(args, evaluator=evaluator)
