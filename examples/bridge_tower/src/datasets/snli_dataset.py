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


class SNLIDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            # names = ["snli_train"]
            names = ["snli_ve_train"]
        elif split == "val":
            # names = ["snli_dev", "snli_test"]  # ViLT, METER
            names = ["snli_ve_dev", "snli_ve_test"]
        elif split == "test":
            names = ["snli_ve_dev", "snli_ve_test"]  # ViLT, METER

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="sentences",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]

        labels = self.table["labels"][index][question_index].as_py()
        return {
            "image": image_tensor,
            "text": text,
            "labels": labels,
            "table_name": self.table_names[index],
        }
