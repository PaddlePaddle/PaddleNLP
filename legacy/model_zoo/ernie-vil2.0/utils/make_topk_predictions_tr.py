# -*- coding: utf-8 -*-

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

"""
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs image-to-text retrieval prediction file for evaluation.
"""

import argparse
import json

import numpy
import numpy as np
import paddle
from tqdm import tqdm

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--image-feats', type=str, required=True, help="Specify the path of image features.")
parser.add_argument('--text-feats', type=str, required=True, help="Specify the path of text features.")
parser.add_argument('--top-k', type=int, default=10, help="Specify the k value of top-k predictions.")
parser.add_argument('--eval-batch-size', type=int, default=32768, help="Specify the image-side batch size when computing the inner products, default to 8192")
parser.add_argument('--output', type=str, required=True, help="Specify the output jsonl prediction filepath.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    print("Begin to load text features...")
    text_ids = []
    text_feats = []
    with open(args.text_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_ids.append(obj["text_id"])
            text_feats.append(obj["feature"])
    text_feats_array = np.array(text_feats, dtype=np.float32)
    print("Finished loading text features.")

    print("Begin to compute top-{} predictions for images...".format(args.top_k))
    with open(args.output, "w") as fout:
        with open(args.image_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                image_id = obj["image_id"]
                image_feat = obj["feature"]
                score_tuples = []
                image_feat_tensor = paddle.to_tensor([image_feat], dtype="float32")  # [1, feature_dim]
                idx = 0
                while idx < len(text_ids):
                    text_feats_tensor = paddle.to_tensor(
                        text_feats_array[idx : min(idx + args.eval_batch_size, len(text_ids))]
                    ).cuda()  # [batch_size, feature_dim]
                    batch_scores = image_feat_tensor @ text_feats_tensor.t()  # [1, batch_size]
                    for text_id, score in zip(
                        text_ids[idx : min(idx + args.eval_batch_size, len(text_ids))],
                        batch_scores.squeeze(0).tolist(),
                    ):
                        score_tuples.append((text_id, score))
                    idx += args.eval_batch_size
                top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[: args.top_k]
                fout.write(
                    "{}\n".format(
                        json.dumps({"image_id": image_id, "text_ids": [entry[0] for entry in top_k_predictions]})
                    )
                )

    print("Top-{} predictions are saved in {}".format(args.top_k, args.output))
    print("Done!")
