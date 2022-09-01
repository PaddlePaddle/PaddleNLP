import numpy as np
import json
from paddlenlp.transformers import GPTTokenizer
import time
import multiprocessing
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='data input/output')
    group.add_argument('-i', '--input_path',
                       type=str,
                       required=True,
                       help='Path to input JSON files.')
    group.add_argument('-o', '--output_prefix',
                       type=str,
                       required=True,
                       help='Output prefix to store output file.')
    group.add_argument(
        '--data_format',
        type=str,
        default='text',
        choices=['JSON'],
        help='Only support json format for now. One document per line.')
    group.add_argument(
        '--json_key',
        type=str,
        default='text',
        help='For JSON format. Space separate listed of keys to extract from json')
    group = parser.add_argument_group(title='common config')
    group.add_argument('--append_eos',
                       action='store_true',
                       help='Append an <eos> token to the end of a document.')
    group.add_argument('--log_interval',
                       type=int,
                       default=100,
                       help='Interval between progress updates')
    group.add_argument('--workers',
                       type=int,
                       default=1,
                       help='Number of worker processes to launch')
    group.add_argument("--vocab_file",
                       type=str,
                       default='./data_tools/code-vocab.json',
                       help="Path to the vocab file")
    group.add_argument("--merge_file",
                       type=str,
                       default='./data_tools/code-merges.txt',
                       help="Path to the BPE merge file (if necessary).")
    return parser.parse_args()


class Sampler(object):
    lang = {'False', 'await', 'else', 'import', 'pass',
            'None', 'break', 'except', 'in', 'raise',
            'True', 'class', 'finally', 'is', 'return',
            'and', 'continue', 'for', 'lambda', 'try',
            'as', 'def', 'from', 'nonlocal', 'while',
            'assert', 'del', 'global', 'not', 'with',
            'async', 'elif', 'if', 'or', 'yield'}
    builtin = {
        'set', 'list', 'dict', 'bool', 'str', 'chr',
        'ord', 'int', 'float', 'format', 'map',
        'filter', 'sum', 'max', 'min', 'mean', 'open',
        'enumerate', 'zip', 'range', 'print', 'input',
        'split', 'self', 'append', 'extend', 'join',
        'pop', 'object', 'match', 'case'
    }

    charset = set([chr(ord('a') + i) for i in range(26)] +
                  [chr(ord('A') + i) for i in range(26)] +
                  ['_'])
    numset = set('.eE-')
    digiset = set('0123456789')
    quo = {"'": 1, '"': 2, "'''": 3, '"""': 4}
    iquo = ['', "'", '"', "'''", '"""']
    ignore = lang | builtin | charset

    nan, idt, num, ostr, anno = 0, 1, 2, 3, 4
    var, func = -1, -2

    def __init__(self, cont, tokenizer, seg_id: int, seq_length: int = 1025):
        # sp: item: [name: str, begin: int, end: int, type: int]
        self.cont, self.sp = '\n' + cont + '\n\n', []
        self.stat = self.nan
        # mp: {identifier: str -> [rank: int, first_appearance: int]}
        self.mp = dict()
        self.tokenizer, self.seg_id, self.seq_length = tokenizer, seg_id, seq_length
        self.err = False
        try:
            self._load()
        except IndexError:
            self.err = True

    @staticmethod
    def builtin(idt: str) -> bool:
        return len(idt) > 4 and idt.startswith('__') and idt.endswith('__')

    def _load(self):
        buf, isf, onf, bkslash, scnt, ecnt, styp = '', False, False, False, 0, 0, ''
        fbrc, bcnt = [(-1, '')], 0

        def _nan(i, x):
            nonlocal buf, onf, scnt, ecnt, styp, bcnt, fbrc
            if x in self.charset:
                self.stat = self.idt
                buf = x
            elif x in self.digiset:
                self.stat = self.num
            elif x in self.quo:
                q = self.cont[i:i + 3]
                styp = q if q[1] == x and q[2] == x else x
                onf, scnt = isf, int(
                    q.startswith("'''") or q.startswith('"""')) << 1
                self.stat = self.ostr
            elif x == '#':
                self.stat = self.anno
            elif x == '}' and bcnt == fbrc[-1][0]:
                styp, onf = fbrc[-1][1], True
                self.stat = self.ostr
                fbrc.pop()
                if not fbrc:
                    raise IndexError

        def _idt(i, x):
            nonlocal buf
            if x in self.charset | self.digiset:
                buf += x
            else:
                self.stat = self.nan
                if x not in self.quo:
                    # left-closed & right-open interval
                    self.sp.append([buf, i - len(buf), i, self.idt])
                buf = ''
                _nan(i, x)

        def _num(i, x):
            if x not in self.numset | self.digiset:
                self.stat = self.nan
                _nan(i, x)

        def _ostr(i, x):
            nonlocal scnt, ecnt, styp
            if scnt:
                scnt -= 1
            elif ecnt:
                ecnt -= 1
                if not ecnt:
                    self.stat = self.nan
            elif onf and x == '{':
                fbrc.append((bcnt + 1, styp))
                self.stat = self.nan
            elif not bkslash and self.cont[i:i + 3].startswith(styp):
                if len(styp) == 1:
                    self.stat = self.nan
                else:
                    ecnt = 2

        def _anno(i, x):
            if x == '\n':
                self.stat = self.nan

        jmp = [_nan, _idt, _num, _ostr, _anno]

        for i, x in enumerate(self.cont):
            jmp[self.stat](i, x)
            isf, bkslash = x == 'f', x == '\\' and not bkslash
            if self.stat != self.anno:
                bcnt += int(x == '{') - int(x == '}')
        if buf and self.stat == self.idt:
            self.sp.append([buf, len(self.cont) - len(buf),
                            len(self.cont), self.idt])
        self.sp.append(
            ['', len(self.cont) + 10, len(self.cont) + 10, self.idt])

    def _filter(self):
        def chk(x) -> bool:
            t = x[0]
            return t not in self.ignore and not self.builtin(t)

        self.sp = list(filter(chk, self.sp))
        for it in self.sp:
            if it[0] in self.mp:
                continue
            self.mp[it[0]] = it[1]

    def _filter2(self):
        seg, cnt, i = [], 0, 0
        ctxt = self.cont.split('\n')
        for line in ctxt:
            cnt += len(line) + 1
            seg.append(cnt)

        vtxt = [list() for _ in range(len(seg))]
        for name, begin, end, _ in self.sp:
            if name in self.ignore or self.builtin(name):
                continue
            if not name:
                break
            while begin >= seg[i]:
                i += 1
            vtxt[i].append(name)
            if name not in self.mp:
                self.mp[name] = i

        return ctxt, vtxt

    def prompt(self, siz: int):
        ctxt, vtxt = self._filter2()
        vlen = np.zeros(len(ctxt))

        head, comma = self.tokenizer('global = '), self.tokenizer(', ')
        length = len(head) + 1
        vbuf, vcnt, vmap = [], 0, dict()
        cbuf = []
        i = len(vlen)
        for i in range(len(vlen) - 1, -1, -1):
            vids = list(map(lambda y: self.tokenizer(y) + comma,
                            filter(lambda x: x not in vmap, vtxt[i])))
            cids = self.tokenizer('\n' + ctxt[i])
            vl = sum(map(len, vids))
            if length + vl - vlen[i] + len(vids) > siz:
                break
            length += vl - vlen[i]
            cbuf.append(cids)
            for j, v in enumerate(vtxt[i]):
                if v not in vmap:
                    ids = self.tokenizer(v) + comma
                    vbuf.append([ids, (i, j)])
                    vmap[v], vcnt = vcnt, vcnt + 1
                    vlen[self.mp[v]] += len(ids)
                    length += len(ids)
                else:
                    vbuf[vmap[v]][1] = (i, j)
        globv = []
        for k, idx in vmap.items():
            if self.mp[k] < i:
                globv.append(vbuf[idx])
        globv.sort(key=lambda x: x[1])
        for v in globv:
            head += v[0]
        idx = len(head)
        for v in reversed(cbuf):
            head += v
        head[idx] = self.seg_id
        return head

    def collect(self):
        self._filter()
        ret = ([], [], [])  # item = ([ids], [i: loss_mask_head], [j: len])
        if self.err:
            return ret
        header, comma = self.tokenizer('global = '), self.tokenizer(
            ', ')  # 'global = ' || 'global = a,'
        hl, cl = len(header), len(comma)
        glob_l, cod_l = hl + 1, 0
        start, cur, j = 0, 0, 0
        buf_c, buf_g, buf_s = [], header[:], set()

        for line in self.cont.split('\n'):
            cur += len(line) + 1
            ids = self.tokenizer(line + '\n')
            cod_l += len(ids)
            buf_c.extend(ids)
            while self.sp[j][1] < cur:
                name = self.sp[j][0]
                if self.mp[name] < start and name not in buf_s:
                    ids = self.tokenizer(name)
                    glob_l += len(ids) + cl
                    buf_g.extend(ids)
                    buf_g.extend(comma)
                    buf_s.add(name)
                j += 1
            if glob_l + cod_l > self.seq_length:
                if glob_l < self.seq_length:
                    buf_g.append(self.seg_id)
                    tmp = buf_g + buf_c[:self.seq_length - glob_l]
                    try:
                        assert len(tmp) == 1025
                    except AssertionError:
                        print(len(tmp), len(buf_g), len(buf_c), glob_l, cod_l)
                        print(line)
                    ret[0].extend(tmp)
                    ret[1].append(len(tmp))
                    ret[2].append(glob_l)
                glob_l, cod_l = hl + 1, 0
                buf_c, buf_g, buf_s = [], header[:], set()
                start = cur + 1
        if cod_l > self.seq_length >> 1 and glob_l < self.seq_length:
            buf_g.append(self.seg_id)
            tmp = buf_g + buf_c
            ret[0].extend(tmp)
            ret[1].append(len(tmp))
            ret[2].append(glob_l)
        return ret


def process(jsonl, key: str, tokenizer, seq_length: int = 1024):
    def tk(x):
        return tokenizer(x)['input_ids']

    return Sampler(json.loads(jsonl)[key], tk,
                   tokenizer.eos_token_id, seq_length + 1).collect()


def prompt_ids(content: str, tokenizer, size: int = 80):
    """
    For example:
    from tio_gen import prompt_ids

    args = get_args()
    tokenizer = GPTTokenizer(args.vocab_file, args.merge_file)
    ids = prompt_ids(args.content, tokenizer, args.size)

    print(tokenizer.convert_ids_to_string(ids))
    """

    def tk(x):
        return tokenizer(x)['input_ids']

    return Sampler(content, tk, tokenizer.eos_token_id).prompt(size)


def test():
    t = time.perf_counter()
    ids = np.load('input_ids_tio.npy')
    npz = np.load('input_idx_tio.npz')
    idx, los = npz['idx'], npz['los']
    print('loaded:', time.perf_counter() - t)

    print('los:', np.mean(los), max(los))
    print('len:', idx[-1] / len(los))

    for i, p in enumerate(los):
        sample = ids[idx[i]:idx[i + 1]]
        # print(eos, sample[p-3:p+3])
        assert sample[p - 1] == eos


if __name__ == '__main__':
    args = get_args()
    seq_len = 1024
    tk = GPTTokenizer(args.vocab_file, args.merge_file)
    eos = tk.eos_token_id

    jl = open(args.input_path, 'r', encoding='utf-8')
    t = time.perf_counter()
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
        ret = p.starmap(process, map(lambda x: (x, 'text', tk, seq_len), jl))
    print('generated, duration =', time.perf_counter() - t)
    jl.close()

    t = time.perf_counter()
    ids, idx, los = [], [0], []
    for s, x, l in ret:
        ids.extend(s)
        idx.extend(x)
        los.extend(l)
    ids = np.array(ids, dtype='int32')
    idx = np.cumsum(np.array(idx, dtype='int64'))
    los = np.array(los)
    with open(args.output_prefix + '_ids_tio.npy', 'wb') as f:
        np.save(f, ids, allow_pickle=True)
    with open(args.output_prefix + '_idx_tio.npz', 'wb') as f:
        np.savez(f, idx=idx, los=los)
    print('saved, duration =', time.perf_counter() - t)
