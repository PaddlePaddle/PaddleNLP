# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from datasets import load_dataset

from .base import RawDataset, RawSample

__all__ = ["JsonDataset"]


class JsonDataset(RawDataset):
    NAME: str = "Jsonfile"

    def __init__(self, path: str | None = None, *args, **kwargs) -> None:
        self.data = load_dataset("json", data_files=path, split="train")
        self.use_rm_server = kwargs.pop("use_rm_server", False)
        assert "src" in self.data.column_names, "'src' should be included in jsonfile"
        if self.use_rm_server:
            assert "tgt" in self.data.column_names, "'tgt' should be included in jsonfile when using rm server"

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        if self.use_rm_server:
            rawdata = RawSample(
                input=data["src"],
                answer=data["tgt"],
            )
        else:
            rawdata = RawSample(input=data["src"])
        return rawdata

    def __len__(self) -> int:
        return len(self.data)  # dataset size
