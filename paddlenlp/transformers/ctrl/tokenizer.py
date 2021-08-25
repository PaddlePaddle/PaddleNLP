import json
import os
import shutil

from paddle.utils import try_import
from .. import PretrainedTokenizer
from paddlenlp.utils.log import logger

__all__ = ['CTRLTokenizer']

CONTROL_CODES = {
    "Pregnancy": 168629,
    "Christianity": 7675,
    "Explain": 106423,
    "Fitness": 63440,
    "Saving": 63163,
    "Ask": 27171,
    "Ass": 95985,
    "Joke": 163509,
    "Questions": 45622,
    "Thoughts": 49605,
    "Retail": 52342,
    "Feminism": 164338,
    "Writing": 11992,
    "Atheism": 192263,
    "Netflix": 48616,
    "Computing": 39639,
    "Opinion": 43213,
    "Alone": 44967,
    "Funny": 58917,
    "Gaming": 40358,
    "Human": 4088,
    "India": 1331,
    "Joker": 77138,
    "Diet": 36206,
    "Legal": 11859,
    "Norman": 4939,
    "Tip": 72689,
    "Weight": 52343,
    "Movies": 46273,
    "Running": 23425,
    "Science": 2090,
    "Horror": 37793,
    "Confession": 60572,
    "Finance": 12250,
    "Politics": 16360,
    "Scary": 191985,
    "Support": 12654,
    "Technologies": 32516,
    "Teenage": 66160,
    "Event": 32769,
    "Learned": 67460,
    "Notion": 182770,
    "Wikipedia": 37583,
    "Books": 6665,
    "Extract": 76050,
    "Confessions": 102701,
    "Conspiracy": 75932,
    "Links": 63674,
    "Narcissus": 150425,
    "Relationship": 54766,
    "Relationships": 134796,
    "Reviews": 41671,
    "News": 4256,
    "Translation": 26820,
    "multilingual": 128406,
}


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class CTRLTokenizer(PretrainedTokenizer):
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }
    ctrl_vocab_link = (
        "http://paddlenlp.bj.bcebos.com/models/transformers/ctrl/vocab.json")
    ctrl_merges_link = (
        "http://paddlenlp.bj.bcebos.com/models/transformers/ctrl/merges.txt")
    pretrained_resource_files_map = {
        "vocab_file": {
            "ctrl": ctrl_vocab_link,
            "sshleifer-tiny-ctrl": ctrl_merges_link,
        },
        "merges_file": {
            "ctrl": ctrl_vocab_link,
            "sshleifer-tiny-ctrl": ctrl_merges_link,
        },
    }
    pretrained_init_configuration = {
        "ctrl": {
            "max_len": 256
        },
        "sshleifer-tiny-ctrl": {
            "max_len": 256
        }
    }

    CONTROL_CODES = CONTROL_CODES

    def __init__(self, vocab_file, merges_file, max_len=None,
                 unk_token="<unk>"):
        self._vocab_file = vocab_file
        self._merges_file = merges_file
        self.max_len = max_len if max_len is not None else int(1e12)

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder)

    def __len__(self):
        return len(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i +
                                                                   1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = "@@ ".join(word)
        word = word[:-4]
        self.cache[token] = word
        return word

    def tokenize(self, text):
        return self._tokenize(text)

    def _tokenize(self, text):
        split_tokens = []
        re = try_import("regex")
        words = re.findall(r"\S+\n?", text)
        for token in words:
            split_tokens.extend([t for t in self.bpe(token).split(" ")])
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    def convert_tokens_to_ids(self, tokens):
        ids = []
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this CTRL model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".
                format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to files under `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            shutil.copyfile(getattr(self, "_%s" % name), save_path)
