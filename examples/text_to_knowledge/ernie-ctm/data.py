import json

from paddlenlp.datasets import MapDataset


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                example = json.loads(line)
                words = example["tokens"]
                tags = example["tags"]
                cls_label = example["cls_label"]
                yield words, tags, cls_label

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            vocab[line.strip()] = i
            i += 1
    return vocab


def convert_example(example,
                    tokenizer,
                    max_seq_len,
                    tags_to_idx=None,
                    labels_to_idx=None,
                    summary_num=2):
    words, tags, cls_label = example
    tokens = ["[CLS%i]" % i for i in range(1, summary_num)] + words
    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    if len(tokenized_input['input_ids']) - 1 - summary_num < len(tags):
        tags = tags[:len(tokenized_input['input_ids']) - 1 - summary_num]
    # '[CLS]' and '[SEP]' will get label 'O'
    tags = ['O'] * (summary_num) + tags + ['O']
    tags += ['O'] * (len(tokenized_input['input_ids']) - len(tags))
    tokenized_input['tags'] = [tags_to_idx[x] for x in tags]
    tokenized_input['cls_label'] = labels_to_idx[cls_label]
    if cls_label in ['编码/引用/列表', '外语句子', '古文/古诗句']:
        tokenized_input['seq_len'] = summary_num
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input[
            'tags'], tokenized_input['cls_label']
