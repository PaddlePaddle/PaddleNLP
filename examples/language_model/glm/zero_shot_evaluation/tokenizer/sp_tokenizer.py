"""
from https://github.com/openai/gpt-2/, changed for chinese
"""
import json
import os
import sentencepiece as spm

"""
SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation 
systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements 
subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the 
extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end 
system that does not depend on language-specific pre/postprocessing.
https://github.com/google/sentencepiece

pip install sentencepiece

or  git clone https://github.com/google/sentencepiece.git
python setup.py install

"""
PRETRAINED_MODEL_FILE = "chinese_sentencepiece/cog-pretrain.model"


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.max_len = 0

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
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
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        return [self.encoder.get(token, 1) for token in self.tokenize(text)]

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        return text

    def tokenize(self, text):
        bpe_tokens = []
        bpe_tokens.extend(bpe_token for bpe_token in self.bpe(text).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.encoder.get(token, 1) for token in tokens]


class Encoder_SP:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text):
        """
        text="...."
        """
        return self.sp.EncodeAsIds(text)

    def decode(self, tokens):
        """
        tokens=[x1,x2,...]
        """
        text = [int(token) for token in tokens]
        # print(text)
        return self.sp.DecodeIds(text)

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        return self.sp.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp.IdToPiece(idx)


def get_encoder(encoder_file, bpe_file):
    # 以下是为了同一个函数入兼容sentencepiece
    filepath, filename = os.path.split(encoder_file)
    shotname, extension = os.path.splitext(filename)

    if (".model" == extension) and (bpe_file == ""):
        return Encoder_SP(encoder_file)
    else:
        with open(encoder_file, 'r', encoding="utf-8") as f:
            encoder = json.load(f)
        with open(bpe_file, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        return Encoder(
            encoder=encoder,
            bpe_merges=bpe_merges,
        )


def from_pretrained():
    return get_encoder(PRETRAINED_MODEL_FILE, "")