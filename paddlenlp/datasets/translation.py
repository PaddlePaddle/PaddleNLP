import os
import io
import collections
import warnings

from functools import partial
import numpy as np

import paddle
from paddle.utils.download import get_path_from_url
from paddlenlp.data import Vocab, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.utils.env import DATA_HOME
from paddle.dataset.common import md5file

__all__ = ['TranslationDataset', 'IWSLT15', 'WMT14ende']


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def get_default_tokenizer():
    """Only support split tokenizer
    """

    def _split_tokenizer(x, delimiter=None):
        if delimiter == "":
            return list(x)
        return x.split(delimiter)

    return _split_tokenizer


class TranslationDataset(paddle.io.Dataset):
    """
    TranslationDataset, provide tuple (source and target) raw data.
    
    Args:
        data(list): Raw data. It is a list of tuple or list, each sample of
            data contains two element, source and target.
    """
    META_INFO = collections.namedtuple('META_INFO', ('src_file', 'tgt_file',
                                                     'src_md5', 'tgt_md5'))
    SPLITS = {}
    URL = None
    MD5 = None
    VOCAB_INFO = None
    UNK_TOKEN = None
    PAD_TOKEN = None
    BOS_TOKEN = None
    EOS_TOKEN = None

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_data(cls, mode="train", root=None):
        """
        Download dataset and read raw data.
        Args:
            mode(str, optional): Data mode to download. It could be 'train',
                'dev' or 'test'. Default: 'train'.
            root (str, optional): Data directory of dataset. If not
                provided, dataset will be saved to default directory
                `~/.paddlenlp/datasets/machine_translation/`. If provided, md5
                check would be performed, and dataset would be downloaded in
                default directory if failed. Default: None.
        Returns:
            list: Raw data, a list of tuple.

        Examples:
            .. code-block:: python

                from paddlenlp.datasets import IWSLT15
                data_path = IWSLT15.get_data()
        """
        root = cls._download_data(mode, root)
        data = cls.read_raw_data(mode, root)
        return data

    @classmethod
    def _download_data(cls, mode="train", root=None):
        """Download dataset"""
        default_root = os.path.join(DATA_HOME, 'machine_translation',
                                    cls.__name__)
        src_filename, tgt_filename, src_data_hash, tgt_data_hash = cls.SPLITS[
            mode]

        filename_list = [
            src_filename, tgt_filename, cls.VOCAB_INFO[0], cls.VOCAB_INFO[1]
        ]
        fullname_list = []
        for filename in filename_list:
            fullname = os.path.join(default_root,
                                    filename) if root is None else os.path.join(
                                        os.path.expanduser(root), filename)
            fullname_list.append(fullname)

        data_hash_list = [
            src_data_hash, tgt_data_hash, cls.VOCAB_INFO[2], cls.VOCAB_INFO[3]
        ]
        for i, fullname in enumerate(fullname_list):
            if not os.path.exists(fullname) or (
                    data_hash_list[i] and
                    not md5file(fullname) == data_hash_list[i]):
                if root is not None:  # not specified, and no need to warn
                    warnings.warn(
                        'md5 check failed for {}, download {} data to {}'.
                        format(filename, cls.__name__, default_root))
                path = get_path_from_url(cls.URL, default_root, cls.MD5)
                return default_root
        return root if root is not None else default_root

    @classmethod
    def get_vocab(cls, root=None):
        """
        Load vocab from vocab files. It vocab files don't exist, the will
        be downloaded.

        Args:
            root (str, optional): Data directory pf dataset. If not provided,
                dataset will be save in `~/.paddlenlp/datasets/machine_translation`.
                If provided, md5 check would be performed, and dataset would be
                downloaded in default directory if failed. Default: None.
        Returns:
            tuple: Source vocab and target vocab.

        Examples:
            .. code-block:: python

                from paddlenlp.datasets import IWSLT15
                (src_vocab, tgt_vocab) = IWSLT15.get_vocab()
        """

        root = cls._download_data(root=root)
        src_vocab_filename, tgt_vocab_filename, _, _ = cls.VOCAB_INFO
        src_file_path = os.path.join(root, src_vocab_filename)
        tgt_file_path = os.path.join(root, tgt_vocab_filename)

        src_vocab = Vocab.load_vocabulary(
            filepath=src_file_path,
            unk_token=cls.UNK_TOKEN,
            pad_token=cls.PAD_TOKEN,
            bos_token=cls.BOS_TOKEN,
            eos_token=cls.EOS_TOKEN)

        tgt_vocab = Vocab.load_vocabulary(
            filepath=tgt_file_path,
            unk_token=cls.UNK_TOKEN,
            pad_token=cls.PAD_TOKEN,
            bos_token=cls.BOS_TOKEN,
            eos_token=cls.EOS_TOKEN)
        return (src_vocab, tgt_vocab)

    @classmethod
    def read_raw_data(cls, mode, root):
        """Read raw data from data files
        Args:
            mode(str): Indicates the mode to read. It could be 'train', 'dev' or
               'test'.
            root(str): Data directory of dataset.
        Returns:
            list: Raw data list.
        """
        src_filename, tgt_filename, _, _ = cls.SPLITS[mode]

        def read_raw_files(corpus_path):
            """Read raw files, return raw data"""
            data = []
            (f_mode, f_encoding, endl) = ("r", "utf-8", "\n")
            with io.open(corpus_path, f_mode, encoding=f_encoding) as f_corpus:
                for line in f_corpus.readlines():
                    data.append(line.strip())
            return data

        src_path = os.path.join(root, src_filename)
        tgt_path = os.path.join(root, tgt_filename)
        src_data = read_raw_files(src_path)
        tgt_data = read_raw_files(tgt_path)

        data = [(src_data[i], tgt_data[i]) for i in range(len(src_data))]
        return data

    @classmethod
    def get_default_transform_func(cls, root=None):
        """Get default transform function, which transforms raw data to id.
        Args:
            root(str, optional): Data directory of dataset.
        Returns:
            tuple: Two transform functions, for source and target data. 
        Examples:
            .. code-block:: python

                from paddlenlp.datasets import IWSLT15
                transform_func = IWSLT15.get_default_transform_func()
        """
        # Get default tokenizer
        src_tokenizer = get_default_tokenizer()
        tgt_tokenizer = get_default_tokenizer()
        src_text_vocab_transform = sequential_transforms(src_tokenizer)
        tgt_text_vocab_transform = sequential_transforms(tgt_tokenizer)

        (src_vocab, tgt_vocab) = cls.get_vocab(root)
        src_text_transform = sequential_transforms(src_text_vocab_transform,
                                                   src_vocab)
        tgt_text_transform = sequential_transforms(tgt_text_vocab_transform,
                                                   tgt_vocab)
        return (src_text_transform, tgt_text_transform)


class IWSLT15(TranslationDataset):
    """
    IWSLT15 Vietnames to English translation dataset.

    Args:
        mode(str, optional): It could be 'train', 'dev' or 'test'. Default: 
            'train'.
        root(str, optional): If None, dataset will be downloaded in default
            directory `~/paddlenlp/datasets/machine_translation/IWSLT15`. If
            provided, md5 check would be performed and dataset would be
            downloaded in default directory if failed. Default: None.
        transform_func(callable, optional): If not None, it transforms raw data
            to index data. Default: None.
    Examples:
        .. code-block:: python

            from paddlenlp.datasets import IWSLT15
            train_dataset = IWSLT15('train')
            train_dataset, valid_dataset = IWSLT15.get_datasets(["train", "dev"])

    """
    URL = "https://paddlenlp.bj.bcebos.com/datasets/iwslt15.en-vi.tar.gz"
    SPLITS = {
        'train': TranslationDataset.META_INFO(
            os.path.join("iwslt15.en-vi", "train.en"),
            os.path.join("iwslt15.en-vi", "train.vi"),
            "5b6300f46160ab5a7a995546d2eeb9e6",
            "858e884484885af5775068140ae85dab"),
        'dev': TranslationDataset.META_INFO(
            os.path.join("iwslt15.en-vi", "tst2012.en"),
            os.path.join("iwslt15.en-vi", "tst2012.vi"),
            "c14a0955ed8b8d6929fdabf4606e3875",
            "dddf990faa149e980b11a36fca4a8898"),
        'test': TranslationDataset.META_INFO(
            os.path.join("iwslt15.en-vi", "tst2013.en"),
            os.path.join("iwslt15.en-vi", "tst2013.vi"),
            "c41c43cb6d3b122c093ee89608ba62bd",
            "a3185b00264620297901b647a4cacf38")
    }
    VOCAB_INFO = (os.path.join("iwslt15.en-vi", "vocab.en"), os.path.join(
        "iwslt15.en-vi", "vocab.vi"), "98b5011e1f579936277a273fd7f4e9b4",
                  "e8b05f8c26008a798073c619236712b4")
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    MD5 = 'aca22dc3f90962e42916dbb36d8f3e8e'

    def __init__(self, mode='train', root=None, transform_func=None):
        data_select = ('train', 'dev', 'test')
        if mode not in data_select:
            raise TypeError(
                '`train`, `dev` or `test` is supported but `{}` is passed in'.
                format(mode))
        if transform_func is not None:
            if len(transform_func) != 2:
                raise ValueError("`transform_func` must have length of two for"
                                 "source and target.")
        # Download data and read data
        self.data = self.get_data(mode=mode, root=root)

        if transform_func is not None:
            self.data = [(transform_func[0](data[0]),
                          transform_func[1](data[1])) for data in self.data]


class WMT14ende(TranslationDataset):
    """
    WMT14 English to German translation dataset.

    Args:
        mode(str, optional): It could be 'train', 'dev' or 'test'. Default: 'train'.
        root(str, optional): If None, dataset will be downloaded in
            `~/.paddlenlp/datasets/machine_translation/WMT14ende/`. If provided,
            md5 check would be performed, and dataset would be downloaded in
            default directory if failed. Default: None.
        transform_func(callable, optional): If not None, it transforms raw data
            to index data. Default: None.
    Examples:
        .. code-block:: python

            from paddlenlp.datasets import WMT14ende
            transform_func = WMT14ende.get_default_transform_func(root=root)
            train_dataset = WMT14ende.get_datasets(mode="train", transform_func=transform_func)
    """
    URL = "https://paddlenlp.bj.bcebos.com/datasets/WMT14.en-de.tar.gz"
    SPLITS = {
        'train': TranslationDataset.META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "train.tok.clean.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "train.tok.clean.bpe.33708.de"),
            "c7c0b77e672fc69f20be182ae37ff62c",
            "1865ece46948fda1209d3b7794770a0a"),
        'dev': TranslationDataset.META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2013.tok.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2013.tok.bpe.33708.de"),
            "aa4228a4bedb6c45d67525fbfbcee75e",
            "9b1eeaff43a6d5e78a381a9b03170501"),
        'test': TranslationDataset.META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2014.tok.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2014.tok.bpe.33708.de"),
            "c9403eacf623c6e2d9e5a1155bdff0b5",
            "0058855b55e37c4acfcb8cffecba1050"),
        'dev-eval': TranslationDataset.META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2013.tok.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2013.tok.de"),
            "d74712eb35578aec022265c439831b0e",
            "6ff76ced35b70e63a61ecec77a1c418f"),
        'test-eval': TranslationDataset.META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2014.tok.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2014.tok.de"),
            "8cce2028e4ca3d4cc039dfd33adbfb43",
            "a1b1f4c47f487253e1ac88947b68b3b8")
    }
    VOCAB_INFO = (os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                               "vocab_all.bpe.33708"),
                  os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                               "vocab_all.bpe.33708"),
                  "2fc775b7df37368e936a8e1f63846bb0",
                  "2fc775b7df37368e936a8e1f63846bb0")
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "<e>"

    MD5 = "a2b8410709ff760a3b40b84bd62dfbd8"

    def __init__(self, mode="train", root=None, transform_func=None):
        if mode not in ("train", "dev", "test", "dev-eval", "test-eval"):
            raise TypeError(
                '`train`, `dev`, `test`, `dev-eval` or `test-eval` is supported but `{}` is passed in'.
                format(mode))
        if transform_func is not None and len(transform_func) != 2:
            if len(transform_func) != 2:
                raise ValueError("`transform_func` must have length of two for"
                                 "source and target.")

        self.data = WMT14ende.get_data(mode=mode, root=root)
        self.mode = mode
        if transform_func is not None:
            self.data = [(transform_func[0](data[0]),
                          transform_func[1](data[1])) for data in self.data]
        super(WMT14ende, self).__init__(self.data)


# For test, not API
def prepare_train_input(insts, pad_id):
    src, src_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[0] for inst in insts])
    tgt, tgt_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[1] for inst in insts])
    return src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis]

batch_size_fn = lambda idx, minibatch_len, size_so_far, data_source: max(size_so_far, len(data_source[idx][0]))

batch_key = lambda size_so_far, minibatch_len: size_so_far * minibatch_len

if __name__ == '__main__':
    batch_size = 4096  #32
    pad_id = 2

    transform_func = IWSLT15.get_default_transform_func()
    train_dataset = IWSLT15(transform_func=transform_func)

    key = (lambda x, data_source: len(data_source[x][0]))

    train_batch_sampler = SamplerHelper(train_dataset).shuffle().sort(
        key=key, buffer_size=batch_size * 20).batch(
            batch_size=batch_size,
            drop_last=True,
            batch_size_fn=batch_size_fn,
            key=batch_key).shard()

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=partial(
            prepare_train_input, pad_id=pad_id))

    for i, data in enumerate(train_loader):
        print(data[1])
        print(paddle.max(data[1]) * len(data[1]))
        print(len(data[1]))
