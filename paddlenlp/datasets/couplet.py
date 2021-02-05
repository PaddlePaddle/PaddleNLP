# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

from paddlenlp.datasets import TranslationDataset

__all__ = ['CoupletDataset']


class CoupletDataset(TranslationDataset):
    """
    Couplet dataset. This dataset is from this github repository:
    https://github.com/v-zich/couplet-clean-dataset, which filters dirty data
    from the original repository https://github.com/wb14123/couplet-dataset.

    Args:
        mode(str, optional): It could be 'train', 'dev' or 'test'. Default: 
            'train'.
        root(str, optional): Data directory of dataset. If not
            provided, dataset will be saved to default directory
            `~/.paddlenlp/datasets/machine_translation/CoupletDataset`. If
            provided, md5 check would be performed, and dataset would be
            downloaded in default directory if failed. Default: None.
    Example:
        .. code-block:: python

            from paddlenlp.datasets import CoupletDataset
            couplet_dataset = CoupletDataset()
    """

    URL = "https://paddlenlp.bj.bcebos.com/datasets/couplet.tar.gz"
    SPLITS = {
        'train': TranslationDataset.META_INFO(
            os.path.join("couplet", "train_src.tsv"),
            os.path.join("couplet", "train_tgt.tsv"),
            "ad137385ad5e264ac4a54fe8c95d1583",
            "daf4dd79dbf26040696eee0d645ef5ad"),
        'dev': TranslationDataset.META_INFO(
            os.path.join("couplet", "dev_src.tsv"),
            os.path.join("couplet", "dev_tgt.tsv"),
            "65bf9e72fa8fdf0482751c1fd6b6833c",
            "3bc3b300b19d170923edfa8491352951"),
        'test': TranslationDataset.META_INFO(
            os.path.join("couplet", "test_src.tsv"),
            os.path.join("couplet", "test_tgt.tsv"),
            "f0a7366dfa0acac884b9f4901aac2cc1",
            "56664bff3f2edfd7a751a55a689f90c2")
    }
    VOCAB_INFO = (os.path.join("couplet", "vocab.txt"), os.path.join(
        "couplet", "vocab.txt"), "0bea1445c7c7fb659b856bb07e54a604",
                  "0bea1445c7c7fb659b856bb07e54a604")
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    MD5 = '5c0dcde8eec6a517492227041c2e2d54'

    def __init__(self, mode='train', root=None):
        data_select = ('train', 'dev', 'test')
        if mode not in data_select:
            raise TypeError(
                '`train`, `dev` or `test` is supported but `{}` is passed in'.
                format(mode))
        # Download and read data
        self.data = self.get_data(mode=mode, root=root)
        self.vocab, _ = self.get_vocab(root)
        self.transform()

    def transform(self):
        eos_id = self.vocab[self.EOS_TOKEN]
        bos_id = self.vocab[self.BOS_TOKEN]
        self.data = [(
            [bos_id] + self.vocab.to_indices(data[0].split("\x02")) + [eos_id],
            [bos_id] + self.vocab.to_indices(data[1].split("\x02")) + [eos_id])
                     for data in self.data]
