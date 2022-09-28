import os

import numpy as np

from paddlenlp.data import Vocab

import paddle
from paddle.io import IterableDataset, DataLoader
import paddle.distributed as dist


class LMDataset(IterableDataset):

    def __init__(self, mode, vocab, path, dataset_name, batch_size, bptt,
                 ext_len, nranks, rank):
        assert (mode
                in ["train", "valid", "test"
                    ]), "Parameter mode must be one of [train, valid, test]."

        super(LMDataset, self).__init__()
        self.vocab = vocab
        self.dataset_name = dataset_name

        if self.dataset_name in ["wt103"]:
            self.data = self.read_raw_data(filename=os.path.join(
                path, mode + ".txt"),
                                           ordered=True,
                                           lower_case=False)
        elif self.dataset_name in ["enwik8", "text8"]:
            self.data = self.read_raw_data(filename=os.path.join(
                path, mode + ".txt"),
                                           ordered=True,
                                           add_eos=False)
        else:
            raise ValueError("Not supported dataset yet. ")
        self.rank = rank
        self.batch_size = batch_size
        batch_size *= nranks

        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.num_step = len(self.data) // batch_size
        data = self.data[:self.num_step * batch_size]
        self.data = data.reshape([batch_size, -1])

        # Number of samples
        self.num_samples = (self.num_step + self.bptt - 1) // self.bptt

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i in range(0, self.data.shape[1] - 1, self.bptt):
            seq_len = min(self.bptt, self.data.shape[1] - 1 - i)
            end_idx = i + seq_len
            beg_idx = max(0, i - self.ext_len)
            src = self.data[:, beg_idx:end_idx]
            target = self.data[:, i + 1:i + 1 + seq_len]

            # NOTE: For now, DataLoader can yield `int`. It's not necessary
            # to transfer `seq_len` after DataLoader.
            # However, if it's necessary to use `seq_len` as input for some
            # PaddlePaddle op, then it must be yielded by `[seq_len]` whose
            # shape is [1], cause some op cannot use shape [] as input.
            yield [
                src[self.rank * self.batch_size:(self.rank + 1) *
                    self.batch_size],
                target[self.rank * self.batch_size:(self.rank + 1) *
                       self.batch_size], seq_len
            ]

    def read_raw_data(self,
                      filename,
                      ordered=False,
                      lower_case=True,
                      delimiter=None,
                      add_eos=True,
                      add_double_eos=False):
        assert os.path.exists(filename), "%s is not exist. " % filename

        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = LMDataset.tokenize(line=line,
                                            delimiter=delimiter,
                                            lower_case=lower_case)
                if add_double_eos:  # for lm1b
                    tokens = [self.vocab._identifiers_to_tokens['bos_token']
                              ] + tokens + [
                                  self.vocab._identifiers_to_tokens['bos_token']
                              ]
                elif add_eos:
                    tokens = tokens + [
                        self.vocab._identifiers_to_tokens['eos_token']
                    ]
                data.append(
                    np.asarray(self.get_indices(tokens)).astype("int64"))

        if ordered:
            data = np.concatenate(data)

        return data

    def get_indices(self, tokens):
        return self.vocab.to_indices(tokens)

    @classmethod
    def get_vocab(cls,
                  files,
                  max_size=None,
                  min_freq=0,
                  lower_case=True,
                  delimiter=None,
                  unk_token=None,
                  pad_token=None,
                  bos_token=None,
                  eos_token=None,
                  **kwargs):
        return Vocab.build_vocab(cls.data_iterator(files=files,
                                                   delimiter=delimiter,
                                                   lower_case=lower_case),
                                 max_size=max_size,
                                 min_freq=min_freq,
                                 unk_token=unk_token,
                                 pad_token=pad_token,
                                 bos_token=bos_token,
                                 eos_token=eos_token)

    @classmethod
    def tokenize(cls, line, delimiter=None, lower_case=True):
        line = line.strip()
        if lower_case:
            line = line.lower()
        tokens = list(line) if delimiter == "" else line.split(delimiter)
        return tokens

    @classmethod
    def data_iterator(cls, files, delimiter=None, lower_case=True):
        if isinstance(files, str):
            files = [files]
        elif not isinstance(files, (list, tuple)):
            raise ValueError(
                "The parameter files must be a str or a list/tuple.")

        for fl in files:
            assert os.path.exists(fl), "%s is not exist. " % fl

            with open(fl, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = cls.tokenize(line=line,
                                          delimiter=delimiter,
                                          lower_case=lower_case)
                    yield tokens


def get_lm_data_loader(args, vocab, mode="train"):
    lm_dataset = LMDataset(
        mode=mode,
        vocab=vocab,
        path=args.data,
        dataset_name=args.dataset,
        batch_size=args.batch_size if mode == "train" else args.eval_batch_size,
        bptt=args.tgt_len,
        ext_len=args.ext_len,
        nranks=dist.get_world_size() if mode == "train" else 1,
        rank=dist.get_rank() if mode == "train" else 0)

    data_loader = DataLoader(dataset=lm_dataset,
                             batch_size=None,
                             num_workers=0,
                             return_list=True)

    return data_loader


def get_lm_vocab(args):
    kwargs = {"unk_token": "<unk>"}
    if args.token_delimiter == "None":
        kwargs["delimiter"] = None
    else:
        kwargs["delimiter"] = args.token_delimiter

    if args.dataset == "wt103":
        kwargs["eos_token"] = "<eos>"
        kwargs["lower_case"] = False

    if args.dataset in ["enwik8", "text8"]:
        files = [
            os.path.join(args.data, "train.txt"),
            os.path.join(args.data, "valid.txt"),
            os.path.join(args.data, "test.txt")
        ]
    elif args.dataset == "wt103":
        files = [os.path.join(args.data, "train.txt")]
    else:
        raise ValueError("Not supported dataset yet. ")

    vocab = LMDataset.get_vocab(files, **kwargs)
    args.ntokens = len(vocab)
    print(
        "Finish processing vocabulary, and the size of vocabulary is {}".format(
            args.ntokens))

    return vocab
