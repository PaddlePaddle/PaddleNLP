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
import tkinter
from tkinter import Label, Tk, PhotoImage, Entry, LEFT, W, END, Button, E
import time
import threading
import json
import uuid
import yaml
from attrdict import AttrDict
import _locale
import jieba
import paddle
from paddlenlp.data import Vocab
from paddlenlp.transformers import position_encoding_init
from paddlenlp.utils.log import logger
from subword_nmt import subword_nmt
import websocket

open_speech = True
try:
    from pyaudio import PyAudio, paInt16
except ImportError as e:
    open_speech = False
    logger.warning("No module named 'pyaudio', so no audio demo.")

import const
from model_demo import SimultaneousTransformerDemo

# By default, the Windows system opens the file with GBK code,
# and the subword_nmt package does not support setting open encoding,
# so it is set to UTF-8 uniformly.
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

is_win = False
if os.name == 'nt':
    is_win = True


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

        self.src_vocab = Vocab.load_vocabulary(args.src_vocab_fpath,
                                               bos_token=args.special_token[0],
                                               eos_token=args.special_token[1],
                                               unk_token=args.special_token[2])

        self.trg_vocab = Vocab.load_vocabulary(args.trg_vocab_fpath,
                                               bos_token=args.special_token[0],
                                               eos_token=args.special_token[1],
                                               unk_token=args.special_token[2])

        args.src_vocab_size = len(self.src_vocab)
        args.trg_vocab_size = len(self.trg_vocab)
        self.args = args

    def tokenize(self, raw_string):
        raw_string = raw_string.strip('\n')
        if not raw_string:
            return raw_string, raw_string
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
                logger.debug('[waitk={}] {}'.format(waitk, res))


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

    if is_win:
        font_align = ('Courier', 15)
        font_label = ('Times', 11)

    window = Tk()

    window.title("Welcome to Simultaneous Translation")
    window.geometry('1200x600')

    logo = PhotoImage(file='images/paddlenlp.png')
    button = Label(window, image=logo, compound='center')
    button.place(x=0, y=0)

    # for chinese input
    lbl1 = Label(window,
                 text="Chinese input:",
                 fg="green",
                 font=font_label,
                 anchor=E,
                 width=28)
    lbl1.place(x=0, y=60)
    txt = Entry(window, font=font_align)
    txt.place(x=250, y=50, width=800, height=50)

    button_on = Button(window, text='REC', relief='raised', cursor="hand2")
    if open_speech:
        button_on.place(x=1090, y=52)

    s_x, s_y = 0, 130
    x, y = 250, 120

    # for jieba+BPE
    lbl2_s = Label(window,
                   text="Jieba+BPE:",
                   fg="black",
                   font=font_label,
                   anchor=E,
                   width=28)
    lbl2_s.place(x=s_x, y=s_y)
    lbl2 = Label(window,
                 text="",
                 font=font_align,
                 background="pale green",
                 anchor=E)
    lbl2.place(x=x, y=y, width=800, height=50)

    # for wait-1
    waitnum = '1'
    lbl3_s = Label(window,
                   text="Simultaneous\nTranslation (wait " + waitnum + "):",
                   fg="red",
                   font=font_label,
                   anchor=E,
                   width=28)
    lbl3_s.place(x=s_x, y=s_y + 70)

    lbl3 = Label(window, text="", font=font_align, background="linen")
    lbl3.place(x=x, y=y + 75, width=800, height=50)

    # for wait-3
    waitnum = '3'
    lbl4_s = Label(window,
                   text="Simultaneous\nTranslation (wait " + waitnum + "):",
                   fg="red",
                   font=font_label,
                   anchor=E,
                   width=28)
    lbl4_s.place(x=s_x, y=s_y + 140)
    lbl4 = Label(window, text="", font=font_align, background="linen")
    lbl4.place(x=x, y=y + 145, width=800, height=50)

    # for wait-5
    waitnum = '5'
    lbl5_s = Label(window,
                   text="Simultaneous\nTranslation (wait " + waitnum + "):",
                   fg="red",
                   font=font_label,
                   anchor=E,
                   width=28)
    lbl5_s.place(x=s_x, y=s_y + 210)
    lbl5 = Label(window, text="", font=font_align, background="linen")
    lbl5.place(x=x, y=y + 215, width=800, height=50)

    # for  wait--1
    lbl6_s = Label(window,
                   text="Full Sentence\nTranslation (wait -1):",
                   fg="blue",
                   font=font_label,
                   anchor=E,
                   width=28)
    lbl6_s.place(x=s_x, y=s_y + 280)

    lbl6 = Label(window, text="", font=font_align, background="sky blue")
    lbl6.place(x=x, y=y + 285, width=800, height=50)

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
            lbl2.configure(text=cut_line(
                (lbl2.cget("text") + ' ' + bpe_str[i]).strip(), 20),
                           fg="black",
                           anchor=W,
                           justify=LEFT)
            window.update()
            if bpe_str[i] in ['。', '？', '！']:
                is_last = True
            translate(args, tokenizer, user_input_tokenized, transformers,
                      waitks, decoder_max_length, is_last, caches, bos_id,
                      all_result)
            lbl3.configure(text=cut_line(
                ' '.join(all_result[0]).replace('@@ ', ''), 11),
                           fg="red",
                           anchor=W,
                           justify=LEFT)
            lbl4.configure(text=cut_line(
                ' '.join(all_result[1]).replace('@@ ', ''), 11),
                           fg="red",
                           anchor=W,
                           justify=LEFT)
            lbl5.configure(text=cut_line(
                ' '.join(all_result[2]).replace('@@ ', ''), 11),
                           fg="red",
                           anchor=W,
                           justify=LEFT)
            lbl6.configure(text=cut_line(
                ' '.join(all_result[3]).replace('@@ ', ''), 11),
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

    def set_val_voice(event=None):
        """
        Start translating
        """

        def send_start_params(ws):
            """
            Send start frame
            :param websocket.WebSocket ws:
            :return:
            """
            req = {
                "type": "START",
                "data": {
                    "appid": const.APPID,
                    "appkey": const.APPKEY,
                    "dev_pid": const.DEV_PID,
                    "cuid": "yourself_defined_user_id",
                    "sample": 16000,
                    "format": "pcm"
                }
            }
            body = json.dumps(req)
            ws.send(body, websocket.ABNF.OPCODE_TEXT)
            logger.info("send START frame with params:" + body)

        def send_audio(ws):
            """
             Send audio
            :param  websocket.WebSocket ws:
            :return:
            """
            # 160ms record
            chunk_ms = 160

            # 160ms *  16000  * 2bytes / 1000ms = 5120bytes
            chunk_len = int(16000 * 2 / 1000 * chunk_ms)

            pa = PyAudio()
            stream = pa.open(format=paInt16,
                             channels=1,
                             rate=16000,
                             input=True,
                             frames_per_buffer=chunk_len // 2)

            while True:
                frames = []
                frame = stream.read(chunk_len // 2, exception_on_overflow=False)
                frames.append(frame)
                body = b''.join(frames)
                if len(body) == 0:
                    logger.info("empty body")
                    continue
                logger.debug("try to send audio length {}".format(len(body)))
                ws.send(body, websocket.ABNF.OPCODE_BINARY)

        def send_finish(ws):
            """
            Send finished frame
            :param websocket.WebSocket ws:
            :return:
            """
            req = {"type": "FINISH"}
            body = json.dumps(req)
            ws.send(body, websocket.ABNF.OPCODE_TEXT)
            logger.info("send FINISH frame")

        def close_websocket(ws_app):
            if ws_app:
                logger.info('close ws_app.')
                send_finish(ws_app)
                ws_app.close()
            logger.info('ws_app closed.')

        def on_open(ws):
            """
            Send data frame after connected
            :param  websocket.WebSocket ws:
            :return:
            """

            def run(*args):
                """
                Send data frame
                :param args:
                :return:
                """
                send_start_params(ws)
                send_audio(ws)
                send_finish(ws)
                logger.debug("thread terminating")

            threading.Thread(target=run).start()

        def on_error(ws, error):
            """
            For error
            :param ws:
            :param error: json
            :return:
                """
            logger.error("error: " + str(error))

        def on_close(ws):
            """
            Close websocket
            :param websocket.WebSocket ws:
            :return:
            """
            logger.info("ws close ...")
            # ws.close()

        def on_message(ws, message):
            """
            Response from server
            :param ws:
            :param message: json
            :return:
            """
            global i
            global text
            global caches
            global bos_id
            global decoder_max_length
            global all_result
            global is_last
            global user_input_bpe
            global user_input_tokenized
            global ws_app
            global start_time

            logger.info("Response: " + message)
            message = json.loads(message)
            if is_last and ws_app:
                close_websocket(ws_app)
            end_time = time.time()
            if end_time - start_time > 10 and ws_app:
                close_websocket(ws_app)
                logger.info(
                    'ws_app started at: {} closed at: {}, cost {}s.'.format(
                        start_time, end_time, end_time - start_time))
            if 'result' in message:
                start_time = time.time()
                text = message['result']
                txt.delete(0, END)
                txt.insert(0, text)
                bpe_str, tokenized_src = tokenizer.tokenize(txt.get())
                while i < len(tokenized_src):
                    user_input_bpe.append(bpe_str[i])
                    user_input_tokenized.append(tokenized_src[i])
                    lbl2.configure(text=cut_line(
                        (lbl2.cget("text") + ' ' + bpe_str[i]).strip(), 20),
                                   fg="black",
                                   anchor=W,
                                   justify=LEFT)
                    window.update()
                    if bpe_str[i] in ['。', '？', '！']:
                        is_last = True
                    translate(args, tokenizer, user_input_tokenized,
                              transformers, waitks, decoder_max_length, is_last,
                              caches, bos_id, all_result)
                    lbl3.configure(text=cut_line(
                        ' '.join(all_result[0]).replace('@@ ', ''), 11),
                                   fg="red",
                                   anchor=W,
                                   justify=LEFT)
                    lbl4.configure(text=cut_line(
                        ' '.join(all_result[1]).replace('@@ ', ''), 11),
                                   fg="red",
                                   anchor=W,
                                   justify=LEFT)
                    lbl5.configure(text=cut_line(
                        ' '.join(all_result[2]).replace('@@ ', ''), 11),
                                   fg="red",
                                   anchor=W,
                                   justify=LEFT)
                    lbl6.configure(text=cut_line(
                        ' '.join(all_result[3]).replace('@@ ', ''), 11),
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
                        if ws_app:
                            close_websocket(ws_app)
                    i += 1

        logger.info("begin")
        uri = const.URI + "?sn=" + str(uuid.uuid1())
        logger.info("uri is " + uri)
        global start_time
        start_time = time.time()
        global ws_app
        ws_app = websocket.WebSocketApp(uri,
                                        on_open=on_open,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
        ws_app.run_forever()

    def clear():
        """
        Clear input and output
        """
        txt.delete(0, END)
        global i
        global text
        global caches
        global bos_id
        global decoder_max_length
        global all_result
        global is_last
        global user_input_bpe
        global user_input_tokenized
        global ws_app
        global start_time
        if ws_app:
            ws_app.close()
        decoder_max_length = [1] * len(waitks)
        caches = [None] * len(waitks)
        bos_id = [None] * len(waitks)
        all_result = [[], [], [], []]
        i = 0
        is_last = False
        user_input_bpe = []
        user_input_tokenized = []
        start_time = 0
        logger.info('CLEAR')
        logger.info(f'i: {i}')
        logger.info(f'caches: {caches}')
        logger.info(f'bos_id: {bos_id}')
        logger.info(f'decoder_max_length: {decoder_max_length}')
        logger.info(f'all_result: {all_result}')
        logger.info(f'is_last: {is_last}')
        lbl2.configure(text="", fg="black", anchor=W, justify=LEFT)
        lbl3.configure(text="", fg="red", anchor=W, justify=LEFT)
        lbl4.configure(text="", fg="red", anchor=W, justify=LEFT)
        lbl5.configure(text="", fg="red", anchor=W, justify=LEFT)
        lbl6.configure(text="", fg="blue", anchor=W, justify=LEFT)
        window.update()

    txt.bind('<Return>', set_val)
    button_on.bind('<Button-1>', set_val_voice)

    desc1 = Label(window,
                  text='使用说明：1. 在Chinese input输入中文，按【回车键】开始实时翻译，'
                  '遇到【。！？】结束整句，按【CLEAR】清空所有的输入和输出；',
                  anchor=E)
    desc1.place(x=s_x + 100, y=s_y + 380)

    backspace_cnt = 19
    if is_win:
        backspace_cnt = 15

    desc2 = Label(window,
                  text=' ' * backspace_cnt + '2. 按【REC】开始录音并开始实时翻译，遇到【。！？】结束整句，'
                  '按【CLEAR】清空所有的输入和输出。',
                  anchor=E)
    if open_speech:
        desc2.place(x=s_x + 100, y=s_y + 410)

    button_clear = Button(window,
                          text='CLEAR',
                          relief="raised",
                          cursor="hand2",
                          command=clear)

    button_clear.place(x=x + 840, y=y + 380)

    window.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
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
        logger.info(f'Loaded wait_{waitk} model.')

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
    # for decoding: is_last=True, max_len=256
    is_last = False
    # subword after bpe
    user_input_bpe = []
    # tokenized id
    user_input_tokenized = []
    # for stream input
    text = ''
    # websocket app
    ws_app = None
    # start time
    start_time = 0

    process(args, tokenizer, transformers, waitks)
