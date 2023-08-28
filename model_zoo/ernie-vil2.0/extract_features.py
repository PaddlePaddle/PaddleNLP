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

import argparse
import json

import paddle
from data_util import get_eval_img_dataset, get_eval_txt_dataset
from paddle.io import DataLoader
from tqdm import tqdm

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import ErnieViLModel, ErnieViLTokenizer

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--extract-image-feats', action="store_true", default=False, help="Whether to extract image features.")
parser.add_argument('--extract-text-feats', action="store_true", default=False, help="Whether to extract text features.")
parser.add_argument('--image-data', type=str, default="../Multimodal_Retrieval/lmdb/test/imgs", help="If --extract-image-feats is True, specify the path of the LMDB directory storing input image base64 strings.")
parser.add_argument('--text-data', type=str, default="../Multimodal_Retrieval/test_texts.jsonl", help="If --extract-text-feats is True, specify the path of input text Jsonl file.")
parser.add_argument('--image-feat-output-path', type=str, default=None, help="If --extract-image-feats is True, specify the path of output image features.")
parser.add_argument('--text-feat-output-path', type=str, default=None, help="If --extract-image-feats is True, specify the path of output text features.")
parser.add_argument("--img-batch-size", type=int, default=64, help="Image batch size.")
parser.add_argument("--text-batch-size", type=int, default=64, help="Text batch size.")
parser.add_argument("--context-length", type=int, default=64, help="The maximum length of input text (include [CLS] & [SEP] tokens).")
parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none)",)
args = parser.parse_args()
# yapf: enable


def main():
    if args.resume is not None:
        # Finetune
        model = ErnieViLModel.from_pretrained(args.resume)
        print("load parameters from " + args.resume)
    else:
        # Zero shot
        model = ErnieViLModel.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")

    # Make inference for texts
    if args.extract_text_feats:
        tokenizer = ErnieViLTokenizer.from_pretrained("ernie_vil-2.0-base-zh")
        eval_dataset = get_eval_txt_dataset(args, tokenizer=tokenizer, max_txt_length=args.context_length)
        my_collate = DataCollatorWithPadding(tokenizer)
        text_loader = DataLoader(eval_dataset, collate_fn=my_collate, batch_size=args.text_batch_size)
        print("Make inference for texts...")
        if args.text_feat_output_path is None:
            args.text_feat_output_path = "{}.txt_feat.jsonl".format(args.text_data[:-6])
        write_cnt = 0
        with open(args.text_feat_output_path, "w") as fout:
            model.eval()
            with paddle.no_grad():
                for batch in tqdm(text_loader):
                    text_ids, texts = batch["text_id"], batch["input_ids"]
                    text_features = model.get_text_features(texts)
                    text_features /= text_features.norm(axis=-1, keepdim=True)
                    for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"text_id": text_id, "feature": text_feature})))
                        write_cnt += 1
        print("{} text features are stored in {}".format(write_cnt, args.text_feat_output_path))

    # Make inference for images
    if args.extract_image_feats:
        image_eval_dataset = get_eval_img_dataset(args)
        image_loader = DataLoader(image_eval_dataset, batch_size=args.img_batch_size)
        print("Make inference for images...")
        if args.image_feat_output_path is None:
            # by default, we store the image features under the same directory with the text features
            args.image_feat_output_path = "{}.img_feat.jsonl".format(args.text_data.replace("_texts.jsonl", "_imgs"))
        write_cnt = 0
        with open(args.image_feat_output_path, "w") as fout:
            model.eval()
            with paddle.no_grad():
                for batch in tqdm(image_loader):
                    image_ids, images = batch
                    image_features = model.get_image_features(pixel_values=images)
                    image_features /= image_features.norm(axis=-1, keepdim=True)
                    if type(image_ids) != list:
                        image_ids = image_ids.tolist()

                    for image_id, image_feature in zip(image_ids, image_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                        write_cnt += 1
        print("{} image features are stored in {}".format(write_cnt, args.image_feat_output_path))

    print("Done!")


if __name__ == "__main__":
    main()
