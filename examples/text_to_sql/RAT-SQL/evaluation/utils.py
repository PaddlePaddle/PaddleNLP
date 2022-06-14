#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import traceback
import logging
import re
import copy
import json

op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!="}
agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
conn_sql_dict = {0: "", 1: "and", 2: "or"}

### from IRNet keywords, need to be simplify
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit',
                   'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

COND_OPS = ('not_in', 'between', '==', '>', '<', '>=', '<=', '!=', 'in', 'like')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

LOGIC_AND_OR = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

CONST_COLUMN = set(['time_now'])

EXPECT_BRACKET_PRE_TOKENS = set(AGG_OPS + SQL_OPS + COND_OPS + CLAUSE_KEYWORDS +
                                ('from', ','))

g_empty_sql = {
    "select": [],
    "from": {
        "conds": [],
        "table_units": []
    },
    "where": [],
    "groupBy": [],
    "having": [],
    "orderBy": [],
    "limit": None,
    "except": None,
    "intersect": None,
    "union": None
}


def is_float(value):
    """is float"""
    try:
        float(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def get_scores(count, pred_total, gold_total):
    """
    Args:

    Returns:
    """
    if pred_total != gold_total:
        return 0, 0, 0
    elif count == pred_total:
        return 1, 1, 1
    return 0, 0, 0


def tokenize_NL2SQL(string, cols, single_equal=False, math=True):
    """
    Args:

    Returns:
    """

    string = string.replace("\'", "\"").lower()
    assert string.count('"') % 2 == 0, "Unexpected quote"

    re_cols = [i.lower() for i in cols]

    def _extract_value(string):
        """extract values in sql"""
        fields = string.split('"')
        for idx, tok in enumerate(fields):
            if idx % 2 == 1:
                fields[idx] = '"%s"' % (tok)
        return fields

    def _resplit(tmp_tokens, fn_split, fn_omit):
        """resplit"""
        new_tokens = []
        for token in tmp_tokens:
            token = token.strip()
            if fn_omit(token):
                new_tokens.append(token)
            elif re.match(r'\d\d\d\d-\d\d(-\d\d)?', token):
                new_tokens.append('"%s"' % (token))
            else:
                new_tokens.extend(fn_split(token))
        return new_tokens

    def _split_aggs(tmp_tokens):
        """split aggs in select"""
        new_toks = []
        for i, tok in enumerate(tmp_tokens):
            if tok in ('from', 'where'):
                new_toks.extend(tmp_tokens[i:])
                break
            if not ((tok.endswith(')') or tok.endswith('),')) and len(tok) > 5):
                new_toks.extend(tok.split(','))
                continue

            extra = ''
            if tok.endswith(','):
                extra = ','
                tok = tok[:-1]

            if tok[:4] in ('sum(', 'avg(', 'max(', 'min('):
                new_toks.extend([tok[:3], '(', tok[4:-1], ')'])
            elif tok[:6] == 'count(':
                new_toks.extend(['count', '(', tok[6:-1], ')'])
            else:
                new_toks.append(tok)

            if extra:
                new_toks.append(extra)

        return new_toks

    def join_by_col(toks, cols):
        new_toks = []
        _len = len(toks)
        i = 0
        while i < _len - 1:
            merge = False
            for j in range(10):
                if ''.join(toks[i:i + j]) in cols:
                    new_toks.append(''.join(toks[i:i + j]))
                    i += j
                    merge = True
            if not merge:
                new_toks.append(toks[i])
                i += 1
        new_toks.append(toks[-1])
        return new_toks

    tokens_tmp = _extract_value(string)

    two_bytes_op = ['==', '!=', '>=', '<=', '<>', '<in>']
    if single_equal:
        if math:
            sep1 = re.compile(r'([ \+\-\*/\(\)=,><;])')  # 单字节运算符
        else:
            sep1 = re.compile(r'([ \(\)=,><;])')
    else:
        if math:
            sep1 = re.compile(r'([ \+\-\*/\(\),><;])')  # 单字节运算符
        else:
            sep1 = re.compile(r'([ \(\),><;])')
    sep2 = re.compile('(' + '|'.join(two_bytes_op) + ')')  # 多字节运算符
    tokens_tmp = _resplit(tokens_tmp, lambda x: x.split(' '),
                          lambda x: x.startswith('"'))
    tokens_tmp = _resplit(tokens_tmp, lambda x: re.split(sep2, x),
                          lambda x: x.startswith('"'))
    tokens_tmp = _split_aggs(tokens_tmp)
    tokens = list(filter(lambda x: x.strip() != '', tokens_tmp))

    tokens = join_by_col(tokens, re_cols)

    def _post_merge(tokens):
        """merge:
              * col name with "(", ")"
              * values with +/-
        """
        idx = 1
        while idx < len(tokens):
            if tokens[idx] == '(' and tokens[
                    idx -
                    1] not in EXPECT_BRACKET_PRE_TOKENS and tokens[idx -
                                                                   1] != '=':
                while idx < len(tokens):
                    tmp_tok = tokens.pop(idx)
                    tokens[idx - 1] += tmp_tok
                    if tmp_tok == ')':
                        break
            elif tokens[idx] in ('+', '-') and tokens[
                    idx - 1] in COND_OPS and idx + 1 < len(tokens):
                tokens[idx] += tokens[idx + 1]
                tokens.pop(idx + 1)
                idx += 1
            else:
                idx += 1
        return tokens

    tokens = _post_merge(tokens)
    if single_equal:
        tokens = [i if i != '=' else '==' for i in tokens]
    return tokens


def sql2query(sql, cols):
    """
    transform sql json to sql query, this is only for NL2SQL, eg. select a, b where a op val1 
    """

    sels = sql['sel']
    aggs = sql['agg']
    op = sql["cond_conn_op"]
    conds = sql["conds"]

    condstrs = [
        f'{cols[cond[0]]} {op_sql_dict[cond[1]]} "{cond[2]}"' for cond in conds
    ]
    cond_str = f" {conn_sql_dict[op]} ".join(condstrs)

    def agg_col(agg, col):
        if agg == 0:
            return cols[col]
        else:
            return f"{agg_sql_dict[agg]} ( {cols[col]} )"

    selstrs = [agg_col(i, j) for i, j in zip(aggs, sels)]
    sel_str = ' , '.join(selstrs)

    return f"SELECT {sel_str} WHERE {cond_str}"


def query2sql(query, cols, single_equal=False, with_value=True):

    cols = [i.lower() for i in cols]

    sql_op_dict = {}
    sql_agg_dict = {}
    sql_conn_dict = {}
    for k, v in op_sql_dict.items():
        sql_op_dict[v] = k
        sql_op_dict[v.lower()] = k
    for k, v in agg_sql_dict.items():
        sql_agg_dict[v] = k
        sql_agg_dict[v.lower()] = k
    for k, v in conn_sql_dict.items():
        sql_conn_dict[v] = k
        sql_conn_dict[v.lower()] = k

    query = tokenize_NL2SQL(query, cols, single_equal=single_equal, math=False)
    assert query[0] == 'select'

    def parse_cols(toks, start_idx):
        """
            :returns next idx, (agg, col)
        """
        if 'from' in toks:
            toks = toks[:toks.index('from')]
        idx = start_idx
        len_ = len(toks)
        outs = []
        while idx < len_:
            if toks[idx] in AGG_OPS:
                agg_id = sql_agg_dict[toks[idx]]
                idx += 1
                assert idx < len_ and toks[idx] == '(', toks[idx]
                idx += 1
                agg, col = toks[start_idx], toks[idx]
                idx += 1
                assert idx < len_ and toks[idx] == ')', toks[idx] + ''.join(
                    toks)
                idx += 1
                outs.append((agg, col))
            elif toks[idx] == ',':
                idx += 1
            else:
                agg, col = '', toks[idx]
                idx += 1
                outs.append(('', col))
        return outs

    def _format_col(old_col):
        """format"""
        if old_col.lower().startswith('table_'):
            return old_col.split('.', 1)[1]
        else:
            return old_col

    if 'where' not in query:
        cond_index = len(query)
        conn = ''
        conds = []
    else:
        cond_index = query.index("where")
        condstr = query[cond_index + 1:]
        conn = [i for i in condstr[3::4]]
        assert len(set(conn)) < 2, conn
        conn = list(set(conn))[0] if conn else ''
        conds = [condstr[i:i + 3] for i in range(len(condstr))[::4]]
    sels = parse_cols(query[:cond_index], 1)

    sql = {}

    sql["agg"] = [sql_agg_dict[i[0]] for i in sels]
    sql["cond_conn_op"] = sql_conn_dict[conn]
    sql["sel"] = [cols.index(_format_col(i[1])) for i in sels]
    if with_value:
        sql["conds"] = [[
            cols.index(_format_col(c[0])), sql_op_dict[c[1]],
            '"' + c[2].strip('\"') + '"'
        ] for c in conds]
    else:
        sql["conds"] = [[cols.index(_format_col(c[0])), sql_op_dict[c[1]], "1"]
                        for c in conds]

    sql_sels = [(sql_agg_dict[i[0]], cols.index(_format_col(i[1])))
                for i in sels]
    return sql, sql_sels


def evaluate_NL2SQL(table, gold, predict, single_equal=False, mode=None):
    scores = {}
    scores_novalue = {}

    # load db
    with open(table) as ifs:
        table_list = json.load(ifs)
        table_dict = {}
        for table in table_list:
            table_dict[table['db_id']] = table

    # load qa
    with open(gold, 'r', encoding='utf-8') as f1, open(predict,
                                                       'r',
                                                       encoding='utf-8') as f2:
        gold_list = [l.strip().split('\t') for l in f1 if len(l.strip()) > 0]
        gold_dict = dict([(x[0], x[1:]) for x in gold_list])

        pred_list = [l.strip().split('\t') for l in f2 if len(l.strip()) > 0]
        pred_dict = dict([(x[0], x[1]) for x in pred_list])

    right = total = 0
    cnt_sel = 0
    cnt_cond = cnt_conn = 0

    def compare_set(gold, pred):
        _pred = copy.deepcopy(pred)
        _gold = copy.deepcopy(gold)

        pred_total = len(_pred)
        gold_total = len(_gold)
        cnt = 0

        for unit in _pred:
            if unit in _gold:
                cnt += 1
                _gold.remove(unit)
        return cnt, pred_total, gold_total

    for qid, item in gold_dict.items():
        total += 1
        if qid not in pred_dict:
            continue
        sql_gold, db_id = ''.join(item[0:-1]), item[-1]

        db = table_dict[db_id]
        cols = [i[1] for i in db["column_names"]]

        sql_pred = pred_dict[qid]

        try:
            sql_gold = sql_gold.replace('==', '=')
            sql_pred = sql_pred.replace('==', '=')
            components_gold, sels_gold = query2sql(sql_gold,
                                                   cols,
                                                   single_equal=single_equal)
            components_pred, sels_pred = query2sql(sql_pred,
                                                   cols,
                                                   single_equal=single_equal)

            cnt, pred_total, gold_total = compare_set(sels_gold, sels_pred)
            score_sels, _, _ = get_scores(cnt, pred_total, gold_total)
            cnt, pred_total, gold_total = compare_set(components_gold["conds"],
                                                      components_pred["conds"])
            score_conds, _, _ = get_scores(cnt, pred_total, gold_total)
            score_conn = components_gold["cond_conn_op"] == components_pred[
                "cond_conn_op"]

            if score_sels:
                cnt_sel += 1
            if score_conds:
                cnt_cond += 1
            if score_conn:
                cnt_conn += 1
            if score_sels and score_conds and score_conn:
                right += 1
            else:
                logging.debug("error instance %s:\npred: %s\ngold: %s" %
                              (qid, sql_pred, sql_gold))
        except Exception as e:
            ##traceback.print_exc()
            logging.warning('parse sql error, error sql:')
            logging.warning(sql_gold + '|||' + sql_pred)
            ##raise e
            continue

    scores["all"] = dict([("count", total), ("exact", right),
                          ("acc", right * 1.0 / total)])
    scores["select"] = dict([("count", total), ("exact", cnt_sel),
                             ("acc", cnt_sel * 1.0 / total)])
    scores["condition"] = dict([("count", total), ("exact", cnt_cond),
                                ("acc", cnt_cond * 1.0 / total)])
    scores["connection"] = dict([("count", total), ("exact", cnt_conn),
                                 ("acc", cnt_conn * 1.0 / total)])

    return scores, scores_novalue


if __name__ == '__main__':
    print(query2sql("SELECT 所在省份 , 产线名称 WHERE 日熔量（吨） < 600", []))
    print(
        query2sql(
            "SELECT MAX ( 货币资金（亿元） ) WHERE 总资产（亿元） > 100 or 净资产（亿元） > 100", []))
    print(
        query2sql("SELECT 股价 , EPS17A WHERE 铁路公司 = 广深铁路",
                  ["股价", "铁路公司", "EPS17A"], True))
    cols = ["公司", "2014（亿元）", "2015（亿元）", "2016（亿元）"]
    print(
        query2sql(
            "SELECT COUNT ( 公司 ) WHERE 2014（亿元） > 20 and 2015（亿元） > 20 and 2016（亿元） > 20",
            cols))

    # print(query2sql("SELECT 书名/Title WHERE 索书号/CallNo. == BF637.U53C555=12010 or ISBN == 9.78142212482e+12", ["书名/Title","索书号/CallNo.",'ISBN']))
    # print(tokenize("SELECT 标称生产企业名称 WHERE 规格(包装规格） == 187.2g/盒 and 标称产品名称 == 富兰克牌西洋参含片", math=False))
    # print(tokenize("SELECT 设备型号 WHERE 生产企业 == AISINAWCO.,LTD. or 设备名称 == WCDMA无线数据终端", math=False))
    # print(tokenize("SELECT sum(t1.amount_claimed) FROM claim_headers AS t1 JOIN claims_documents AS t2 ON t1.claim_header_id  =  t2.claim_id WHERE t2.created_date  =  ( SELECT created_date FROM claims_documents ORDER BY created_date LIMIT 1 )"))
    # print(query2sql("SELECT 书号（ISBN) WHERE 教材名称 == 线性代数 or 教材名称 == 中级有机化学", ["书号（ISBN)", "教材名称" ]))
