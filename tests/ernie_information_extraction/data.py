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

from paddlenlp.datasets import MapDataset


def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            key = line.strip('\n')
            vocab[key] = i
            i += 1
    return vocab


def load_dataset(datafiles):

    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def parse_decodes(sentences, predictions, lengths, label_vocab):
    """Parse the padding result

    Args:
        sentences (list): the tagging sentences.
        predictions (list): the prediction tags.
        lengths (list): the valid length of each sentence.
        label_vocab (dict): the label vocab.

    Returns:
        outputs (list): the formatted output.
    """
    predictions = [x for batch in predictions for x in batch]
    lengths = [x for batch in lengths for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lengths):
        sent = sentences[idx][:end]
        tags = [id_label[x] for x in predictions[idx][:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs
