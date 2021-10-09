# coding:utf-8
import sys
from paddlenlp.transformers import (
    RobertaModel, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification)
from paddlenlp.transformers import RobertaBPETokenizer, RobertaTokenizer
import paddle
import os
import numpy as np


def decode(start, end, topk, max_answer_len, undesired_tokens):
    """
        Take the output of any :obj:`ModelForQuestionAnswering` and will generate probabilities for each span to be the
        actual answer.
        """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]
    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
    desired_spans = np.isin(starts, undesired_tokens.nonzero()) & np.isin(
        ends, undesired_tokens.nonzero())
    starts = starts[desired_spans]
    ends = ends[desired_spans]
    scores = candidates[0, starts, ends]

    return starts, ends, scores


def compare_roberta_base(model_name='roberta-base'):
    path = 'paddlenlp/transformers/roberta/{}'.format(model_name.split('/')[-1])

    model = RobertaForMaskedLM.from_pretrained(path)
    tokenizer = RobertaBPETokenizer.from_pretrained(model_name)
    text = ["The man worked as a", "."]  #"The man worked as a <mask>."
    tokens_list = []
    for i in range(2):
        tokens_list.append(tokenizer.tokenize(text[i]))

    tokens = ['<s>']
    tokens.extend(tokens_list[0])
    tokens.extend(['<mask>'])
    tokens.extend(tokens_list[1])
    tokens.extend(['</s>'])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(token_ids)

    model.eval()
    input_ids = paddle.to_tensor([token_ids])
    with paddle.no_grad():
        pd_outputs = model(input_ids)

    pd_outputs = pd_outputs[0]

    pd_outputs_sentence = "paddle: "
    for i, id in enumerate(token_ids):
        if id == 50264:
            scores, index = paddle.nn.functional.softmax(pd_outputs[i],
                                                         -1).topk(5)
            tokens = tokenizer.convert_ids_to_tokens(index.tolist())
            outputs = []
            for score, tk in zip(scores.tolist(), tokens):
                outputs.append(f"{tk}={score}")
            pd_outputs_sentence += "[" + "||".join(outputs) + "]" + " "
        else:
            pd_outputs_sentence += "".join(
                tokenizer.convert_ids_to_tokens(
                    [id], skip_special_tokens=True)) + " "

    print(pd_outputs_sentence)


def compare_deepsort_qa():
    path = 'paddlenlp/transformers/roberta/roberta-base-squad2'
    tokenizer = RobertaBPETokenizer.from_pretrained("roberta-base")
    questions = ['Where do I live?']
    contexts = ['My name is Sarah and I live in London']

    token = tokenizer(
        questions,
        contexts,
        stride=128,
        max_seq_len=64,
        return_attention_mask=True,
        return_special_tokens_mask=True)
    # print(token)
    special_tokens_mask = token[0]['special_tokens_mask']
    count = 3
    st_idx = 0
    for i in special_tokens_mask:
        st_idx += 1
        if i == 1:
            count -= 1
        if count == 0:
            break

    input_ids = token[0]['input_ids']
    offset_mapping = token[0]['offset_mapping']

    input_ids = paddle.to_tensor(input_ids, dtype='int64').unsqueeze(0)

    model = RobertaForQuestionAnswering.from_pretrained(path)
    model.eval()
    start, end = model(input_ids=input_ids)
    start_ = start[0].numpy()
    end_ = end[0].numpy()
    undesired_tokens = np.ones_like(input_ids[0].numpy())

    undesired_tokens[1:st_idx] = 0
    undesired_tokens[-1] = 0

    # Generate mask
    undesired_tokens_mask = undesired_tokens == 0.0

    # Make sure non-context indexes in the tensor cannot contribute to the softmax
    start_ = np.where(undesired_tokens_mask, -10000.0, start_)
    end_ = np.where(undesired_tokens_mask, -10000.0, end_)

    start_ = np.exp(start_ - np.log(
        np.sum(np.exp(start_), axis=-1, keepdims=True)))
    end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
    start_idx, end_idx, score = decode(start_, end_, 1, 64, undesired_tokens)
    start_idx, end_idx = offset_mapping[start_idx[0]][0], offset_mapping[
        end_idx[0]][1]
    print("ans: {}".format(contexts[0][start_idx:end_idx]),
          'score:{}'.format(score.item()))


def compare_uer_text_cls():
    tokenizer = RobertaTokenizer.from_pretrained(
        "uer/roberta-base-finetuned-chinanews-chinese")
    token = tokenizer('北京上个月召开了两会')
    # print(token)
    path = 'paddlenlp/transformers/roberta/roberta-base-finetuned-chinanews-chinese'
    config = RobertaModel.pretrained_init_configuration[
        'uer/roberta-base-finetuned-chinanews-chinese']
    roberta = RobertaModel(**config)
    model = RobertaForSequenceClassification(roberta, 7)
    model_state = paddle.load(os.path.join(path, "model_state.pdparams"))
    model.load_dict(model_state)
    model.eval()
    input_ids = paddle.to_tensor(token['input_ids'], dtype='int64').unsqueeze(0)
    with paddle.no_grad():
        output = model(input_ids)
    import paddle.nn.functional as F
    output = F.softmax(output)
    id2label = {
        "0": "mainland China politics",
        "1": "Hong Kong - Macau politics",
        "2": "International news",
        "3": "financial news",
        "4": "culture",
        "5": "entertainment",
        "6": "sports"
    }
    for i in range(7):
        print("{}: \t {}".format(id2label[str(i)], output[0][i].item()))


def compare_uer_clue_ner():
    text = '江苏警方通报特斯拉冲进店铺'
    tokenizer = RobertaTokenizer.from_pretrained(
        "uer/roberta-base-finetuned-cluener2020-chinese")
    token = tokenizer(text)

    path = 'paddlenlp/transformers/roberta/roberta-base-finetuned-cluener2020-chinese'
    config = RobertaModel.pretrained_init_configuration[
        'uer/roberta-base-finetuned-cluener2020-chinese']
    roberta = RobertaModel(**config)
    model = RobertaForTokenClassification(roberta, 32)
    model_state = paddle.load(os.path.join(path, "model_state.pdparams"))
    model.load_dict(model_state)
    model.eval()
    input_ids = paddle.to_tensor(token['input_ids'], dtype='int64').unsqueeze(0)
    with paddle.no_grad():
        output = model(input_ids)

    import paddle.nn.functional as F
    output = F.softmax(output)
    id2label = {
        "0": "O",
        "1": "B-address",
        "2": "I-address",
        "3": "B-book",
        "4": "I-book",
        "5": "B-company",
        "6": "I-company",
        "7": "B-game",
        "8": "I-game",
        "9": "B-government",
        "10": "I-government",
        "11": "B-movie",
        "12": "I-movie",
        "13": "B-name",
        "14": "I-name",
        "15": "B-organization",
        "16": "I-organization",
        "17": "B-position",
        "18": "I-position",
        "19": "B-scene",
        "20": "I-scene",
        "21": "S-address",
        "22": "S-book",
        "23": "S-company",
        "24": "S-game",
        "25": "S-government",
        "26": "S-movie",
        "27": "S-name",
        "28": "S-organization",
        "29": "S-position",
        "30": "S-scene",
        "31": "[PAD]"
    }
    tokenized_text = tokenizer.tokenize(text)
    scores = []
    char_cn = []
    for t, s in zip(tokenized_text, output[0][1:-1]):
        index = paddle.argmax(s).item()
        label = id2label[str(index)]
        if index != 0:
            scores.append(s[index].item())
            char_cn.append(t)
            print(f"{label} {t} score {s[index].item()}")
    print("{}:{}".format("".join(char_cn[:2]), sum(scores[:2]) / 2))
    print("{}:{}".format("".join(char_cn[2:]), sum(scores[2:]) / 3))


def compare_uer_qa():
    tokenizer = RobertaTokenizer.from_pretrained(
        "uer/roberta-base-chinese-extractive-qa")
    questions = ['著名诗歌《假如生活欺骗了你》的作者是']
    contexts = [
        '普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。'
    ]
    token = tokenizer(questions, contexts, stride=256, max_seq_len=256)
    token_type_ids = token[0]["token_type_ids"]
    offset_mapping = token[0]["offset_mapping"]
    st_idx = len(token_type_ids) - sum(token_type_ids)

    path = 'paddlenlp/transformers/roberta/roberta-base-chinese-extractive-qa'
    model = RobertaForQuestionAnswering.from_pretrained(path)
    model.eval()
    input_ids = paddle.to_tensor(
        token[0]['input_ids'], dtype='int64').unsqueeze(0)
    token_type_ids = paddle.to_tensor(
        token[0]['token_type_ids'], dtype='int64').unsqueeze(0)
    with paddle.no_grad():
        start, end = model(input_ids, token_type_ids)
    start_ = start[0].numpy()
    end_ = end[0].numpy()
    undesired_tokens = np.ones_like(input_ids[0].numpy())

    undesired_tokens[1:st_idx] = 0
    undesired_tokens[-1] = 0

    # Generate mask
    undesired_tokens_mask = undesired_tokens == 0.0

    # Make sure non-context indexes in the tensor cannot contribute to the softmax
    start_ = np.where(undesired_tokens_mask, -10000.0, start_)
    end_ = np.where(undesired_tokens_mask, -10000.0, end_)

    start_ = np.exp(start_ - np.log(
        np.sum(np.exp(start_), axis=-1, keepdims=True)))
    end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
    start_idx, end_idx, score = decode(start_, end_, 1, 64, undesired_tokens)
    start_idx, end_idx = offset_mapping[start_idx[0]][0], offset_mapping[
        end_idx[0]][1]
    print("ans: {}".format(contexts[0][start_idx:end_idx]),
          'score:{}'.format(score.item()))


if __name__ == "__main__":
    # compare_deepsort_qa()
    """
    ### paddlenlp
    ans: London score:0.7771294116973877
    ###huggingface
    https://huggingface.co/deepset/roberta-base-squad2?context=My+name+is+Sarah+and+I+live+in+London&question=Where+do+I+live%3F
    {
    "score": 0.777230978012085,
    "start": 31,
    "end": 37,
    "answer": "London"
    }
    """
    compare_uer_text_cls()
    """
    ### paddlenlp:
    mainland China politics:         0.7211663722991943
    Hong Kong - Macau politics:      0.0015174386790022254
    International news:      0.0003004991449415684
    financial news:          0.19014376401901245
    culture:         0.07077095657587051
    entertainment:   0.007615785114467144
    sports:          0.008485240861773491

    ### huggingface:
    https://huggingface.co/uer/roberta-base-finetuned-chinanews-chinese?text=%E5%8C%97%E4%BA%AC%E4%B8%8A%E4%B8%AA%E6%9C%88%E5%8F%AC%E5%BC%80%E4%BA%86%E4%B8%A4%E4%BC%9A
    [
        {
        "label": "mainland China politics",
        "score": 0.7211663126945496
        },
        {
        "label": "Hong Kong - Macau politics",
        "score": 0.0015174405416473746
        },
        {
        "label": "International news",
        "score": 0.0003004991449415684
        },
        {
        "label": "financial news",
        "score": 0.19014358520507812
        },
        {
        "label": "culture",
        "score": 0.07077110558748245
        },
        {
        "label": "entertainment",
        "score": 0.007615773007273674
        },
        {
        "label": "sports",
        "score": 0.008485235273838043
        }
    ]
    """

    compare_uer_clue_ner()
    """
    ### paddlenlp:
    B-address 江 score 0.6619006395339966
    I-address 苏 score 0.5544567704200745
    B-company 特 score 0.42272716760635376
    I-company 斯 score 0.45469826459884644
    I-company 拉 score 0.5207838416099548
    江苏:0.6081787049770355
    特斯拉:0.466069757938385
    ### hugging_face:
    https://huggingface.co/uer/roberta-base-finetuned-cluener2020-chinese?text=%E6%B1%9F%E8%8B%8F%E8%AD%A6%E6%96%B9%E9%80%9A%E6%8A%A5%E7%89%B9%E6%96%AF%E6%8B%89%E5%86%B2%E8%BF%9B%E5%BA%97%E9%93%BA
    {
        "entity_group": "address",
        "score": 0.6081769466400146,
        "word": "江 苏",
        "start": 0,
        "end": 2
    },
    {
        "entity_group": "company",
        "score": 0.466068834066391,
        "word": "特 斯 拉",
        "start": 6,
        "end": 9
    }
    """
    compare_uer_qa()
    """
    paddlenlp:
    ans: 普希金 score:0.9766426682472229
    huggingface:
    {
    "score": 0.9766426086425781,
    "start": 0,
    "end": 3,
    "answer": "普希金"
    }
    """
    compare_roberta_base('roberta-base')
    """
    ### paddlenlp:
    paddle:  The Ġman Ġworked Ġas Ġa [Ġmechanic=0.08702456951141357||Ġwaiter=0.08196529746055603||Ġbutcher=0.07332348823547363||Ġminer=0.04632236436009407||Ġguard=0.04015030339360237] .

    """
    compare_roberta_base('roberta-large')
    """
    ### paddlenlp:
    paddle:  The Ġman Ġworked Ġas Ġa [Ġmechanic=0.08259474486112595||Ġdriver=0.057356804609298706||Ġteacher=0.04709123820066452||Ġbartender=0.046422384679317474||Ġwaiter=0.042401134967803955] .
    
    ### huggingface:
    [
    {
        "sequence": "The man worked as a mechanic.",
        "score": 0.08260270208120346,
        "token": 25682,
        "token_str": " mechanic"
    },
    {
        "sequence": "The man worked as a driver.",
        "score": 0.057361237704753876,
        "token": 1393,
        "token_str": " driver"
    },
    {
        "sequence": "The man worked as a teacher.",
        "score": 0.04709038510918617,
        "token": 3254,
        "token_str": " teacher"
    },
    {
        "sequence": "The man worked as a bartender.",
        "score": 0.04641640931367874,
        "token": 33080,
        "token_str": " bartender"
    },
    {
        "sequence": "The man worked as a waiter.",
        "score": 0.04239244386553764,
        "token": 38233,
        "token_str": " waiter"
    }
    ]
    """
    compare_roberta_base('sshleifer/tiny-distilroberta-base')
    """
    ### paddlenlp:
    paddle:  The Ġman Ġworked Ġas Ġa [ĠMaul=2.2346663172356784e-05||ELS=2.234388557553757e-05||Ġirregular=2.2234522475628182e-05||ĠDr=2.2128468117443845e-05||³=2.2114423700259067e-05] .
    
    ### huggingface:
    [
        {
            "sequence": "The man worked as a mechanic.",
            "score": 0.08260270208120346,
            "token": 25682,
            "token_str": " mechanic"
        },
        {
            "sequence": "The man worked as a driver.",
            "score": 0.057361237704753876,
            "token": 1393,
            "token_str": " driver"
        },
        {
            "sequence": "The man worked as a teacher.",
            "score": 0.04709038510918617,
            "token": 3254,
            "token_str": " teacher"
        },
        {
            "sequence": "The man worked as a bartender.",
            "score": 0.04641640931367874,
            "token": 33080,
            "token_str": " bartender"
        },
        {
            "sequence": "The man worked as a waiter.",
            "score": 0.04239244386553764,
            "token": 38233,
            "token_str": " waiter"
        }
    ]
    """
