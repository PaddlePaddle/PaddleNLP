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
"""model utils file"""
import json
import os

MODEL_FILE = "luke.ckpt"
METADATA_FILE = "metadata.json"
TSV_ENTITY_VOCAB_FILE = "entity_vocab.tsv"
ENTITY_VOCAB_FILE = "entity_vocab.jsonl"


def get_entity_vocab_file_path(directory: str) -> str:
    """get entity vocab file"""
    default_entity_vocab_file_path = os.path.join(directory, ENTITY_VOCAB_FILE)
    tsv_entity_vocab_file_path = os.path.join(directory, TSV_ENTITY_VOCAB_FILE)

    if os.path.exists(tsv_entity_vocab_file_path):
        return tsv_entity_vocab_file_path
    if os.path.exists(default_entity_vocab_file_path):
        return default_entity_vocab_file_path
    raise FileNotFoundError(f"{directory} does not contain any entity vocab files.")


class ModelArchive:
    """model archive"""

    def __init__(self, model_path, metadata: dict):
        """init fun"""
        self.model_path = model_path
        self.metadata = metadata

    @property
    def bert_model_name(self):
        """bert model name"""
        return self.metadata["model_config"]["bert_model_name"]

    @property
    def max_seq_length(self):
        """max seq len"""
        return self.metadata["max_seq_length"]

    @property
    def max_mention_length(self):
        """max mention len"""
        return self.metadata["max_mention_length"]

    @property
    def max_entity_length(self):
        """max entity len"""
        return self.metadata["max_entity_length"]

    @classmethod
    def load(cls, archive_path: str):
        """load fun"""
        return cls._load(archive_path)

    @staticmethod
    def _load(path: str):
        """load fun"""
        model_path = os.path.join(path, MODEL_FILE)
        metadata = json.load(open(os.path.join(path, METADATA_FILE)))
        return ModelArchive(model_path, metadata)
