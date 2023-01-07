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

from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            # names = ["vqav2_train", "vqav2_trainable_val"] # ViLT
            names = ["vqav2_train", "vqav2_val"]  # METER
            # We fix a bug in ViLT & METER's write_vqa.py.
            # names = ["vqav2_train_fix", "vqav2_val_fix"] # METER_fix
            # names = ["vqav2_train_fix", "vqav2_val_fix", "vgqa_coco_train", "vgqa_coco_val"] # + vgqa coco only
            # names = ["vqav2_train_fix", "vqav2_val_fix", "vgqa_train", "vgqa_val"] # + vgqa all
        elif split == "val":
            # names = ["vqav2_rest_val"] # ViLT
            names = ["vqav2_val"]  # METER
            # names = ["vqav2_val_fix"] # METER_fix
        elif split == "test":
            names = ["vqav2_test"]  # evaluate the test-dev and test-std
            # names = ["vqav2_test-dev"]  # only evaluate the test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            scores = list()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }
