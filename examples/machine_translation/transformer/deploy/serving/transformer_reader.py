import numpy as np

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Pad, Vocab


class TransformerReader(object):

    def __init__(self, args={}):
        super(TransformerReader, self).__init__()

        dataset = load_dataset('wmt14ende', splits=('test'))
        if not args.benchmark:
            self.vocab = Vocab.load_vocabulary(**dataset.vocab_info["bpe"])
        else:
            self.vocab = Vocab.load_vocabulary(
                **dataset.vocab_info["benchmark"])
        self.src_vocab = self.trg_vocab = self.vocab

        def convert_samples(samples):
            source = []
            for sample in samples:
                src = sample.split()
                source.append(self.src_vocab.to_indices(src))

            return source

        self.tokenize = convert_samples
        self.to_tokens = self.trg_vocab.to_tokens
        self.feed_keys = ["src_word"]
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.pad_idx = args.bos_idx
        self.pad_seq = args.pad_seq
        self.word_pad = Pad(self.pad_idx)

    def set_feed_keys(self, keys):
        self.feed_keys = keys

    def get_feed_keys(self):
        return self.feed_keys

    def prepare_infer_input(self, insts):
        """
        Put all padded data needed by beam search decoder into a list.
        """
        insts = self.tokenize(insts)

        src_max_len = (max([len(inst) for inst in insts]) +
                       self.pad_seq) // self.pad_seq * self.pad_seq
        src_word = self.word_pad([
            inst + [self.eos_idx] + [self.pad_idx] *
            (src_max_len - 1 - len(inst)) for inst in insts
        ])

        return np.asarray(src_word)
