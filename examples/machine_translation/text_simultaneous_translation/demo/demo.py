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
import os
import argparse
from tkinter import Label, Tk, PhotoImage, Entry, LEFT, W, END, Button, N, E
import yaml
from attrdict import AttrDict

import _locale
import jieba
import paddle
from paddlenlp.data import Vocab
from paddlenlp.transformers import position_encoding_init
from subword_nmt import subword_nmt

from model_for_demo import SimultaneousTransformerDemo

# By default, the Windows system opens the file with GBK code,
# and the subword_nmt package does not support setting open encoding,
# so it is set to UTF-8 uniformly.
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])


class STACLTokenizer:
    """
    Jieba+BPE, and convert tokens to ids.
    """

    def __init__(self, args, is_chinese):
        bpe_parser = subword_nmt.create_apply_bpe_parser()
        bpe_args = bpe_parser.parse_args(args=['-c', args.src_bpe_dict])
        self.bpe = subword_nmt.BPE(bpe_args.codes, bpe_args.merges,
                                   bpe_args.separator, None,
                                   bpe_args.glossaries)
        self.is_chinese = is_chinese

        self.src_vocab = Vocab.load_vocabulary(
            args.src_vocab_fpath,
            bos_token=args.special_token[0],
            eos_token=args.special_token[1],
            unk_token=args.special_token[2])

        self.trg_vocab = Vocab.load_vocabulary(
            args.trg_vocab_fpath,
            bos_token=args.special_token[0],
            eos_token=args.special_token[1],
            unk_token=args.special_token[2])

        args.src_vocab_size = len(self.src_vocab)
        args.trg_vocab_size = len(self.trg_vocab)
        self.args = args

    def tokenize(self, raw_string):
        raw_string = raw_string.strip('\n')
        if not raw_string:
            return raw_string
        if self.is_chinese:
            raw_string = ' '.join(jieba.cut(raw_string))
        bpe_str = self.bpe.process_line(raw_string)
        ids = self.src_vocab.to_indices(bpe_str.split())
        return bpe_str.split(), ids


def init_model(args, init_from_params):
    # Define model
    args.init_from_params = init_from_params
    transformer = SimultaneousTransformerDemo(
        args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
        args.n_layer, args.n_head, args.d_model, args.d_inner_hid, args.dropout,
        args.weight_sharing, args.bos_idx, args.eos_idx, args.waitk)

    # Load the trained model
    assert args.init_from_params, (
        "Please set init_from_params to load the infer model.")

    model_dict = paddle.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))

    # To avoid a longer length than training, reset the size of position
    # encoding to max_length
    model_dict["src_pos_embedding.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    model_dict["trg_pos_embedding.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)

    transformer.load_dict(model_dict)
    return transformer


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def translate(args, tokenizer, tokenized_src, transformers, waitks,
              decoder_max_length, is_last, caches, bos_id, all_result):
    # Set evaluate mode
    for transformer in transformers:
        transformer.eval()

    for idx, (waitk, transformer) in enumerate(zip(waitks, transformers)):
        if len(tokenized_src) < waitk or (waitk == -1 and not is_last):
            print()
            continue
        with paddle.no_grad():
            input_src = tokenized_src
            if is_last:
                decoder_max_length[idx] = args.max_out_len
                input_src += [args.eos_idx]
            src_word = paddle.to_tensor(input_src).unsqueeze(axis=0)
            finished_seq, finished_scores, cache = transformer.greedy_search(
                src_word,
                max_len=decoder_max_length[idx],
                waitk=waitk,
                caches=caches[idx],
                bos_id=bos_id[idx])
            caches[idx] = cache
            finished_seq = finished_seq.numpy()
            finished_scores = finished_scores.numpy()
            for beam_idx, beam in enumerate(finished_seq[0]):
                if beam_idx >= args.n_best:
                    break
                id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                if len(id_list) == 0:
                    continue
                bos_id[idx] = id_list[-1]
                word_list = tokenizer.trg_vocab.to_tokens(id_list)
                for word in word_list:
                    all_result[idx].append(word)
                res = ' '.join(word_list).replace('@@ ', '')
                print(res)
                print()


def cut_line(str, line_len):
    """
    Wrap output
    """
    result = []
    temp = []
    for idx, item in enumerate(str.split()):
        temp.append(item)
        if (idx + 1) % line_len == 0:
            result.append(' '.join(temp))
            temp = []
    if len(temp) != 0:
        result.append(' '.join(temp))
    return '\n'.join(result)


def process(args, tokenizer, transformers, waitks):
    """
    GUI and main waitk program
    :param args:
    :param tokenizer:
    :param transformers:
    :param waitks:
    :return:
    """
    font_align = ('Courier', 20)
    font_label = ('Times', 14)

    window = Tk()

    window.title("Welcome to Simultaneous Translation(text)")
    window.geometry('1500x600')

    logo = PhotoImage(file='images/paddlenlp.png')
    button = Label(window, image=logo, compound='center')
    button.grid(column=0, row=0)

    # for chinese input
    lbl1 = Label(
        window,
        text="Chinese input:",
        fg="green",
        font=font_label,
        anchor=E,
        width=28,
        justify=LEFT)
    lbl1.grid(column=0, row=1)
    txt = Entry(window, width=130, font=('Courier', 24))
    txt.grid(column=1, row=1)
    txt.focus()

    # for jieba+BPE
    lbl2 = Label(
        window,
        text="",
        width=150,
        font=font_align,
        background="pale green",
        anchor=E,
        justify=LEFT)
    lbl2.grid(column=1, row=2)
    lbl2_s = Label(
        window,
        text="Jieba+BPE:",
        fg="black",
        font=font_label,
        anchor=E,
        width=28)
    lbl2_s.grid(column=0, row=2)

    # for wait-1
    lbl3 = Label(
        window, text="", width=150, font=font_align, background="linen")
    lbl3.grid(column=1, row=3)
    waitnum = '1'
    lbl3_s = Label(
        window,
        text="Simultaneous\nTranslation (wait " + waitnum + "):",
        fg="red",
        font=font_label,
        anchor=E,
        width=28)
    lbl3_s.grid(column=0, row=3)

    # for wait-3
    lbl4 = Label(
        window, text="", width=150, font=font_align, background="linen")
    lbl4.grid(column=1, row=4)
    waitnum = '3'
    lbl4_s = Label(
        window,
        text="Simultaneous\nTranslation (wait " + waitnum + "):",
        fg="red",
        font=font_label,
        anchor=E,
        width=28)
    lbl4_s.grid(column=0, row=4)

    # for wait-5
    lbl5 = Label(
        window, text="", width=150, font=font_align, background="linen")
    lbl5.grid(column=1, row=5)
    waitnum = '5'
    lbl5_s = Label(
        window,
        text="Simultaneous\nTranslation (wait " + waitnum + "):",
        fg="red",
        font=font_label,
        anchor=E,
        width=28)
    lbl5_s.grid(column=0, row=5)

    # for  wait--1
    lbl6 = Label(
        window, text="", width=150, font=font_align, background="sky blue")
    lbl6.grid(column=1, row=6)
    lbl6_s = Label(
        window,
        text="Full Sentence\nTranslation (wait -1):",
        fg="blue",
        font=font_label,
        anchor=E,
        width=28)
    lbl6_s.grid(column=0, row=6)

    def set_val(event=None):
        """
        Start translating
        """
        global i
        global caches
        global bos_id
        global decoder_max_length
        global all_result
        global is_last
        global user_input_bpe
        global user_input_tokenized
        bpe_str, tokenized_src = tokenizer.tokenize(txt.get())
        while i < len(tokenized_src):
            user_input_bpe.append(bpe_str[i])
            user_input_tokenized.append(tokenized_src[i])
            lbl2.configure(
                text=cut_line((lbl2.cget("text") + ' ' + bpe_str[i]).strip(),
                              20),
                fg="black",
                anchor=W,
                justify=LEFT)
            window.update()
            if bpe_str[i] in ['。', '？', '！']:
                is_last = True
            translate(args, tokenizer, user_input_tokenized, transformers,
                      waitks, decoder_max_length, is_last, caches, bos_id,
                      all_result)
            lbl3.configure(
                text=cut_line(' '.join(all_result[0]).replace('@@ ', ''), 14),
                fg="red",
                anchor=W,
                justify=LEFT)
            lbl4.configure(
                text=cut_line(' '.join(all_result[1]).replace('@@ ', ''), 14),
                fg="red",
                anchor=W,
                justify=LEFT)
            lbl5.configure(
                text=cut_line(' '.join(all_result[2]).replace('@@ ', ''), 14),
                fg="red",
                anchor=W,
                justify=LEFT)
            lbl6.configure(
                text=cut_line(' '.join(all_result[3]).replace('@@ ', ''), 14),
                fg="blue",
                anchor=W,
                justify=LEFT)
            window.update()
            if is_last:
                caches = [None] * len(waitks)
                bos_id = [None] * len(waitks)
                decoder_max_length = [1] * len(waitks)
                is_last = False
                user_input_bpe = []
                user_input_tokenized = []
            i += 1

    def clear():
        """
        Clear input and output
        """
        txt.delete(0, END)
        global i
        global caches
        global bos_id
        global decoder_max_length
        global all_result
        global is_last
        global user_input_bpe
        global user_input_tokenized
        decoder_max_length = [1] * len(waitks)
        caches = [None] * len(waitks)
        bos_id = [None] * len(waitks)
        all_result = [[], [], [], []]
        i = 0
        is_last = False
        user_input_bpe = []
        user_input_tokenized = []
        print('CLEAR')
        print(f'i: {i}')
        print(f'caches: {caches}')
        print(f'bos_id: {bos_id}')
        print(f'decoder_max_length: {decoder_max_length}')
        print(f'all_result: {all_result}')
        print(f'is_last: {is_last}')
        lbl2.configure(text="", fg="black", anchor=W, justify=LEFT)
        lbl3.configure(text="", fg="red", anchor=W, justify=LEFT)
        lbl4.configure(text="", fg="red", anchor=W, justify=LEFT)
        lbl5.configure(text="", fg="red", anchor=W, justify=LEFT)
        lbl6.configure(text="", fg="blue", anchor=W, justify=LEFT)
        window.update()

    txt.bind('<Return>', set_val)

    button = Label(window, height=2)
    button.grid(column=1, row=7)

    image = PhotoImage(file='images/clear.png')
    button1 = Button(
        window,
        image=image,
        width=60,
        height=15,
        relief="ridge",
        cursor="hand2",
        command=clear)
    button1.grid(column=1, row=8, sticky=N)

    button = Label(
        window,
        text='使用说明：在Chinese input输入中文，按【回车键】开始实时翻译，'
        '遇到【。！？】结束整句，按【CLEAR】清空所有的输入和输出。' + ' ' * 300,
        anchor=E,
        justify=LEFT)
    button.grid(
        column=0, row=9, rowspan=2, columnspan=2, padx=10, ipadx=10, pady=10)

    window.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./transformer_demo.yaml",
        type=str,
        help="Path of the config file. ")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    yaml_file = args.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))

    if args.device == 'gpu':
        place = "gpu:0"
    elif args.device == 'xpu':
        place = "xpu:0"
    elif args.device == 'cpu':
        place = "cpu"
    paddle.set_device(place)

    tokenizer = STACLTokenizer(args, is_chinese=True)
    waitks = [1, 3, 5, -1]

    transformers = []
    for waitk in waitks:
        transformers.append(init_model(args, f'models/nist_wait_{waitk}'))
        print(f'Loaded wait_{waitk} model.')

    # for decoding max length
    decoder_max_length = [1] * len(waitks)
    # for decoding cache
    caches = [None] * len(waitks)
    # for decoding start token id
    bos_id = [None] * len(waitks)
    # for result
    all_result = [[], [], [], []]
    # current source word index
    i = 0
    # for decoding: is_last=True,max_len=256 
    is_last = False
    # subword after bpe
    user_input_bpe = []
    # tokenized id
    user_input_tokenized = []

    process(args, tokenizer, transformers, waitks)
