#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from args import parse_args
from dataset_cmrc2018 import get_dev_dataloader
from train_cmrc2018 import MODEL_CLASSES
from tqdm.auto import tqdm
from metric import compute_prediction
from utils import save_json
import os


@paddle.no_grad()
def evaluate(model, data_loader, args, output_dir="./"):
    model.eval()
    all_start_logits = []
    all_end_logits = []

    for batch in tqdm(data_loader):
        input_ids, token_type_ids, pinyin_ids = batch
        start_logits_tensor, end_logits_tensor = model(
            input_ids, token_type_ids=token_type_ids, pinyin_ids=pinyin_ids)
        all_start_logits.extend(start_logits_tensor.numpy().tolist())
        all_end_logits.extend(end_logits_tensor.numpy().tolist())

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        data_loader.dataset.data,
        data_loader.dataset.new_data,
        (all_start_logits, all_end_logits),
        False,
        args.n_best_size,
        args.max_answer_length,
        args.null_score_diff_threshold, )

    save_json(all_predictions, os.path.join(output_dir, "all_predictions.json"))
    if args.save_nbest_json:
        save_json(all_nbest_json,
                  os.path.join(output_dir, "all_nbest_json.json"))


def main(args):
    print(args)
    paddle.set_device(args.device)
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    splits = "test"
    dev_data_loader = get_dev_dataloader(tokenizer, args, splits=splits)
    evaluate(model, dev_data_loader, args, output_dir=args.output_dir)

    data_dir = args.data_dir
    dev_ground_truth_file_path = os.path.join(data_dir, "dev.json")
    dev_predict_file_path = os.path.join(args.output_dir,
                                         "all_predictions.json")
    if splits == "dev":
        from cmrc_evaluate import get_result
        get_result(dev_ground_truth_file_path, dev_predict_file_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
