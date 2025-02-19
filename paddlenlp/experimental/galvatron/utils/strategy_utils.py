import numpy as np

def form_strategy(strategy):
    template = '%d-%s-%s'
    assert len(strategy) == 4
    info = strategy[-1]
    pp_deg = strategy[0]
    tp_deg = '%d'%strategy[1]
    dp_deg = '%d'%strategy[2]
    if 'fsdp' in info.keys():
        if info['fsdp']:
            dp_deg += 'f'
    if 'tp' in info.keys():
        if info['tp']:
            tp_deg += '*'
        else:
            dp_deg += '*'
    if 'cpt' in info.keys():
        if info['cpt']:
            dp_deg += '-c'
    
    if 'sp' in info.keys():
        if info['sp']:
            dp_deg += '-sp'
    
    return template%(pp_deg, tp_deg, dp_deg)

def strategy_str2list(strategy_str):
    s = strategy_str.split('-')
    if '*' in s[1]:
        tp_consec = 1
        s[1] = s[1][:-1]
    elif '*' in s[2]:
        tp_consec = 0
        s[2] = s[2][:-1]
    if 'f' in s[2]:
        fsdp = 1
        s[2] = s[2][:-1]
    else:
        fsdp = 0
    if len(s) >= 4 and s[3] == 'c':
        cpt = 1
    else:
        cpt = 0
    if len(s) >= 5 and s[4] == 'sp':
        sp = 1
    else:
        sp = 0
    
    pp_deg, tp_deg, dp_deg = int(s[0]), int(s[1]), int(s[2])
    re = [pp_deg, tp_deg, dp_deg, {}]
    if tp_deg > 1 and dp_deg > 1:
        re[-1]['tp'] = tp_consec
    if dp_deg > 1:
        re[-1]['fsdp'] = fsdp
    if cpt == 1:
        re[-1]['cpt'] = 1
    if sp == 1:
        re[-1]['sp'] = 1
    return re

def print_strategies(strategy_list):
    if strategy_list is None or isinstance(strategy_list, str):
        print(None)
        return
    if isinstance(strategy_list[0][0],list):
        result_list = []
        for sub_strategy_list in strategy_list:
            sub_result_list = []
            for strategy in sub_strategy_list:
                sub_result_list.append(form_strategy(strategy))
            result_list.append(', '.join(sub_result_list))
        print(' || '.join(result_list))
    else:
        result_list = []
        for strategy in strategy_list:
            result_list.append(form_strategy(strategy))
        print(', '.join(result_list))