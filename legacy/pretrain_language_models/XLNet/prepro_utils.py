# coding=utf-8
"""this file is a copy of https://github.com/zihangdai/xlnet"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata
import six
from functools import partial

SPIECE_UNDERLINE = '▁'


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def print_(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, list):
            s = [printable_text(i) for i in arg]
            s = ' '.join(s)
            new_args.append(s)
        else:
            new_args.append(printable_text(arg))
    print(*new_args)


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if six.PY2 and isinstance(outputs, str):
        outputs = outputs.decode('utf-8')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    # return_unicode is used only for py2

    # note(zhiliny): in some systems, sentencepiece only accepts str for py2
    if six.PY2 and isinstance(text, unicode):
        text = text.encode('utf-8')

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(piece[:-1].replace(
                SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][
                    0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = piece.decode('utf-8')
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


if __name__ == '__main__':
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load('sp10m.uncased.v3.model')

    print_(u'I was born in 2000, and this is falsé.')
    print_(u'ORIGINAL',
           sp.EncodeAsPieces(u'I was born in 2000, and this is falsé.'))
    print_(u'OURS',
           encode_pieces(sp, u'I was born in 2000, and this is falsé.'))
    print(encode_ids(sp, u'I was born in 2000, and this is falsé.'))
    print_('')
    prepro_func = partial(preprocess_text, lower=True)
    print_(prepro_func('I was born in 2000, and this is falsé.'))
    print_('ORIGINAL',
           sp.EncodeAsPieces(
               prepro_func('I was born in 2000, and this is falsé.')))
    print_('OURS',
           encode_pieces(sp,
                         prepro_func('I was born in 2000, and this is falsé.')))
    print(encode_ids(sp, prepro_func('I was born in 2000, and this is falsé.')))
    print_('')
    print_('I was born in 2000, and this is falsé.')
    print_('ORIGINAL',
           sp.EncodeAsPieces('I was born in 2000, and this is falsé.'))
    print_('OURS', encode_pieces(sp, 'I was born in 2000, and this is falsé.'))
    print(encode_ids(sp, 'I was born in 2000, and this is falsé.'))
    print_('')
    print_('I was born in 92000, and this is falsé.')
    print_('ORIGINAL',
           sp.EncodeAsPieces('I was born in 92000, and this is falsé.'))
    print_('OURS', encode_pieces(sp, 'I was born in 92000, and this is falsé.'))
    print(encode_ids(sp, 'I was born in 92000, and this is falsé.'))
