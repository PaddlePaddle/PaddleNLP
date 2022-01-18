# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


def reset_offset(pred_words):
    for i in range(0, len(pred_words)):
        if i > 0:
            pred_words[i]["offset"] = pred_words[i - 1]["offset"] + len(
                pred_words[i - 1]["item"])
        pred_words[i]["length"] = len(pred_words[i]["item"])
    return pred_words


def decode(texts, all_pred_tags, summary_num, idx_to_tags):
    batch_results = []
    for i, pred_tags in enumerate(all_pred_tags):
        pred_words, pred_word = [], []

        for j, tag in enumerate(pred_tags[summary_num:-1]):
            if j >= len(texts[i]):
                break
            pred_label = idx_to_tags[tag]
            if pred_label.find("-") != -1:
                _, label = pred_label.split("-")
            else:
                label = pred_label
            if pred_label.startswith("S") or pred_label.startswith("O"):
                pred_words.append({
                    "item": texts[i][j],
                    "offset": 0,
                    "wordtag_label": label
                })
            else:
                pred_word.append(texts[i][j])
                if pred_label.startswith("E"):
                    pred_words.append({
                        "item": "".join(pred_word),
                        "offset": 0,
                        "wordtag_label": label
                    })
                    del pred_word[:]

        pred_words = reset_offset(pred_words)
        result = {"text": texts[i], "items": pred_words}
        batch_results.append(result)
    return batch_results
