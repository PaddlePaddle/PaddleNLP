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


class CocoCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            # names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"] # ViLT
            names = ["coco_karpathy_train", "coco_karpathy_restval"]  # ViLT
        elif split == "val":
            # names = ["coco_caption_karpathy_val"]
            # names = ["coco_caption_karpathy_test"] # ViLT, METER
            names = ["coco_karpathy_test"]  # ViLT, METER
        elif split == "test":
            names = ["coco_karpathy_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split:
            _index, _question_index = self.index_mapper[index]
            iid = self.table["image_id"][_index].as_py()
            iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite
