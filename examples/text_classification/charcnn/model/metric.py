#!/usr/bin/env python3
from termcolor import cprint, colored as c

def inc(d, label):
    if label in d:
        d[label] += 1
    else:
        d[label] = 1

def precision_recall(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))
    labels = set(target)
    TP = {}
    TP_plus_FN = {}
    TP_plus_FP = {}
    for i in range(len(output)):

        inc(TP_plus_FN, target[i])
        inc(TP_plus_FP, output[i])
        if target[i] == output[i]:
            inc(TP, output[i])

    for label in labels:
        if label not in TP_plus_FN:
            TP_plus_FN[label] = 0
        if label not in TP_plus_FP:
            TP_plus_FP[label] = 0

    precision = {label: 0. if TP_plus_FP[label] ==0 else ((TP[label] if label in TP else 0) / float(TP_plus_FP[label])) for label in labels}
    recall = {label: 0. if TP_plus_FN[label] ==0 else ((TP[label] if label in TP else 0) / float(TP_plus_FN[label])) for label in labels}

    return precision, recall, TP, TP_plus_FN, TP_plus_FP


def F_score(p, r):

    f_scores = {
        label: None if p[label] == 0 and r[label] == 0 else (0 if p[label] == 0 or r[label] == 0 else 2 / (1 / p[label] + 1 / r[label]))
        for label in p
    }
    return f_scores


def print_f_score(output, target):
    """returns: 
        p<recision>, 
        r<ecall>, 
        f<-score>, 
        {"TP", "p", "TP_plus_FP"} """
    p, r, TP, TP_plus_FN, TP_plus_FP = precision_recall(output, target)
    f = F_score(p, r)


    for label in f.keys():
        cprint("Label: " + c(("  " + str(label))[-5:], 'red') +
               "\tPrec: " + c("  {:.1f}".format(p[label] * 100)[-5:], 'green') + '%' +
               " ({:d}/{:d})".format((TP[label] if label in TP else 0), TP_plus_FP[label]).ljust(14) +
               "Recall: " + c("  {:.1f}".format((r[label] if label in r else 0) * 100)[-5:], 'green') + "%" +
               " ({:d}/{:d})".format((TP[label] if label in TP else 0), TP_plus_FN[label]).ljust(14) +
               "F-Score: " + ("  N/A" if f[label] is None else (c("  {:.1f}".format(f[label] * 100)[-5:], "green") + "%"))
               )
