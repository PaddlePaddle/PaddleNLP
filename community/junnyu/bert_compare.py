from paddlenlp.transformers import BertForSequenceClassification, BertForMaskedLM, BertForTokenClassification, BertTokenizer
import paddle


def compare_math(path="MODEL/tbs17-MathBERT"):
    model = BertForMaskedLM.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(path)
    model.eval()
    print()
    text = "The man worked as a [MASK]."
    tokens = ["[CLS]"]
    text_list = text.split("[MASK]")
    for i, t in enumerate(text_list):
        tokens.extend(tokenizer.tokenize(t))
        if i == len(text_list) - 1:
            tokens.extend(["[SEP]"])
        else:
            tokens.extend(["[MASK]"])

    input_ids_list = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = paddle.to_tensor([input_ids_list])
    with paddle.no_grad():
        pd_outputs = model(input_ids)[0]
    pd_outputs_sentence = "paddle: "
    for i, id in enumerate(input_ids_list):
        if id == tokenizer.convert_tokens_to_ids(["[MASK]"])[0]:
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

    # paddle: the man worked as a [book=0.6469274759292603||guide=0.07073356211185455||text=0.031362663954496384||man=0.023064589127898216||distance=0.02054688334465027] .  
    # expected:
    # https://huggingface.co/tbs17/MathBERT
    # [{'score': 0.6469377875328064,
    # 'sequence': 'the man worked as a book.',
    # 'token': 2338,
    # 'token_str': 'book'},
    #  {'score': 0.07073448598384857,
    # 'sequence': 'the man worked as a guide.',
    # 'token': 5009,
    # 'token_str': 'guide'},
    #  {'score': 0.031362924724817276,
    # 'sequence': 'the man worked as a text.',
    # 'token': 3793,
    # 'token_str': 'text'},
    #  {'score': 0.02306508645415306,
    # 'sequence': 'the man worked as a man.',
    # 'token': 2158,
    # 'token_str': 'man'},
    #  {'score': 0.020547250285744667,
    # 'sequence': 'the man worked as a distance.',
    # 'token': 3292,
    # 'token_str': 'distance'}]


def compare_nlptown(
        path="MODEL/nlptown-bert-base-multilingual-uncased-sentiment"):
    model = BertForSequenceClassification.from_pretrained(path)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(path)
    text = "I like you. I love you"
    inputs = {
        k: paddle.to_tensor(
            v, dtype="int64").unsqueeze(0)
        for k, v in tokenizer(text).items()
    }
    with paddle.no_grad():
        score = paddle.nn.functional.softmax(model(**inputs), axis=-1)
    id2label = {
        0: "1 star",
        1: "2 stars",
        2: "3 stars",
        3: "4 stars",
        4: "5 stars"
    }
    for i, s in enumerate(score[0].tolist()):
        label = id2label[i]
        print(f"{label} score {s}")

    # 1 star score 0.0021950288210064173
    # 2 stars score 0.0022533712908625603
    # 3 stars score 0.015475980937480927
    # 4 stars score 0.1935628354549408
    # 5 stars score 0.7865128517150879
    # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you
    # 1 star 0.002
    # 2 stars 0.002
    # 3 stars 0.015
    # 4 stars 0.194
    # 5 stars 0.787


def compare_ckiplab_ws(path="MODEL/ckiplab-bert-base-chinese-ws"):
    model = BertForTokenClassification.from_pretrained(path)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(path)
    text = "我叫克拉拉，我住在加州伯克利。"
    tokenized_text = tokenizer.tokenize(text)
    inputs = {
        k: paddle.to_tensor(
            v, dtype="int64").unsqueeze(0)
        for k, v in tokenizer(text).items()
    }
    with paddle.no_grad():
        score = paddle.nn.functional.softmax(model(**inputs), axis=-1)
    id2label = {0: "B", 1: "I"}
    for t, s in zip(tokenized_text, score[0][1:-1]):
        index = paddle.argmax(s).item()
        label = id2label[index]
        print(f"{label} {t} score {s[index].item()}")

    # B 我 score 0.9999921321868896
    # B 叫 score 0.9999772310256958
    # B 克 score 0.9999295473098755
    # I 拉 score 0.999772846698761
    # I 拉 score 0.9999483823776245
    # B ， score 0.9999879598617554
    # B 我 score 0.9999914169311523
    # B 住 score 0.9999860525131226
    # B 在 score 0.6059999465942383
    # B 加 score 0.9999884366989136
    # I 州 score 0.9999697208404541
    # B 伯 score 0.999879002571106
    # I 克 score 0.9999772310256958
    # I 利 score 0.9999678134918213
    # B 。 score 0.9999856948852539

    # https://huggingface.co/ckiplab/bert-base-chinese-ws
    # [
    # {
    #     "entity_group": "B",
    #     "score": 0.9999921321868896,
    #     "word": "我",
    #     "start": 0,
    #     "end": 1
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.9999772906303406,
    #     "word": "叫",
    #     "start": 1,
    #     "end": 2
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.9999296069145203,
    #     "word": "克",
    #     "start": 2,
    #     "end": 3
    # },
    # {
    #     "entity_group": "I",
    #     "score": 0.999772846698761,
    #     "word": "拉",
    #     "start": 3,
    #     "end": 4
    # },
    # {
    #     "entity_group": "I",
    #     "score": 0.9999484419822693,
    #     "word": "拉",
    #     "start": 4,
    #     "end": 5
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.9999879598617554,
    #     "word": "，",
    #     "start": 5,
    #     "end": 6
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.9999914169311523,
    #     "word": "我",
    #     "start": 6,
    #     "end": 7
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.9999861121177673,
    #     "word": "住",
    #     "start": 7,
    #     "end": 8
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.6059996485710144,
    #     "word": "在",
    #     "start": 8,
    #     "end": 9
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.999988317489624,
    #     "word": "加",
    #     "start": 9,
    #     "end": 10
    # },
    # {
    #     "entity_group": "I",
    #     "score": 0.9999697804450989,
    #     "word": "州",
    #     "start": 10,
    #     "end": 11
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.9998790621757507,
    #     "word": "伯",
    #     "start": 11,
    #     "end": 12
    # },
    # {
    #     "entity_group": "I",
    #     "score": 0.9999772310256958,
    #     "word": "克",
    #     "start": 12,
    #     "end": 13
    # },
    # {
    #     "entity_group": "I",
    #     "score": 0.9999678134918213,
    #     "word": "利",
    #     "start": 13,
    #     "end": 14
    # },
    # {
    #     "entity_group": "B",
    #     "score": 0.9999856352806091,
    #     "word": "。",
    #     "start": 14,
    #     "end": 15
    # }
    # ]


def compare_ckiplab_pos(path="MODEL/ckiplab-bert-base-chinese-pos"):
    model = BertForTokenClassification.from_pretrained(path)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(path)
    text = "我叫沃尔夫冈，我住在柏林。"
    tokenized_text = tokenizer.tokenize(text)
    inputs = {
        k: paddle.to_tensor(
            v, dtype="int64").unsqueeze(0)
        for k, v in tokenizer(text).items()
    }
    with paddle.no_grad():
        score = paddle.nn.functional.softmax(model(**inputs), axis=-1)
    id2label = {
        "0": "A",
        "1": "Caa",
        "2": "Cab",
        "3": "Cba",
        "4": "Cbb",
        "5": "D",
        "6": "Da",
        "7": "Dfa",
        "8": "Dfb",
        "9": "Di",
        "10": "Dk",
        "11": "DM",
        "12": "I",
        "13": "Na",
        "14": "Nb",
        "15": "Nc",
        "16": "Ncd",
        "17": "Nd",
        "18": "Nep",
        "19": "Neqa",
        "20": "Neqb",
        "21": "Nes",
        "22": "Neu",
        "23": "Nf",
        "24": "Ng",
        "25": "Nh",
        "26": "Nv",
        "27": "P",
        "28": "T",
        "29": "VA",
        "30": "VAC",
        "31": "VB",
        "32": "VC",
        "33": "VCL",
        "34": "VD",
        "35": "VF",
        "36": "VE",
        "37": "VG",
        "38": "VH",
        "39": "VHC",
        "40": "VI",
        "41": "VJ",
        "42": "VK",
        "43": "VL",
        "44": "V_2",
        "45": "DE",
        "46": "SHI",
        "47": "FW",
        "48": "COLONCATEGORY",
        "49": "COMMACATEGORY",
        "50": "DASHCATEGORY",
        "51": "DOTCATEGORY",
        "52": "ETCCATEGORY",
        "53": "EXCLAMATIONCATEGORY",
        "54": "PARENTHESISCATEGORY",
        "55": "PAUSECATEGORY",
        "56": "PERIODCATEGORY",
        "57": "QUESTIONCATEGORY",
        "58": "SEMICOLONCATEGORY",
        "59": "SPCHANGECATEGORY"
    }
    for t, s in zip(tokenized_text, score[0][1:-1]):
        index = paddle.argmax(s).item()
        label = id2label[str(index)]
        print(f"{label} {t} score {s[index].item()}")

    # Nh 我 score 1.0
    # VG 叫 score 0.9999830722808838
    # Nc 沃 score 0.9999146461486816
    # Nc 尔 score 0.9999760389328003
    # Nc 夫 score 0.9984875917434692
    # Na 冈 score 0.8717513680458069
    # COMMACATEGORY ， score 1.0
    # Nh 我 score 1.0
    # VCL 住 score 0.9999992847442627
    # P 在 score 0.9999998807907104
    # Nc 柏 score 0.9999998807907104
    # Nc 林 score 0.9891127943992615
    # PERIODCATEGORY 。 score 1.0

    # https://huggingface.co/ckiplab/bert-base-chinese-pos
    # [
    # {
    #     "entity_group": "Nh",
    #     "score": 1,
    #     "word": "我",
    #     "start": 0,
    #     "end": 1
    # },
    # {
    #     "entity_group": "VG",
    #     "score": 0.9999829530715942,
    #     "word": "叫",
    #     "start": 1,
    #     "end": 2
    # },
    # {
    #     "entity_group": "Nc",
    #     "score": 0.9999146461486816,
    #     "word": "沃",
    #     "start": 2,
    #     "end": 3
    # },
    # {
    #     "entity_group": "Nc",
    #     "score": 0.9999760389328003,
    #     "word": "尔",
    #     "start": 3,
    #     "end": 4
    # },
    # {
    #     "entity_group": "Nc",
    #     "score": 0.9984875917434692,
    #     "word": "夫",
    #     "start": 4,
    #     "end": 5
    # },
    # {
    #     "entity_group": "Na",
    #     "score": 0.8717513084411621,
    #     "word": "冈",
    #     "start": 5,
    #     "end": 6
    # },
    # {
    #     "entity_group": "COMMACATEGORY",
    #     "score": 1,
    #     "word": "，",
    #     "start": 6,
    #     "end": 7
    # },
    # {
    #     "entity_group": "Nh",
    #     "score": 1,
    #     "word": "我",
    #     "start": 7,
    #     "end": 8
    # },
    # {
    #     "entity_group": "VCL",
    #     "score": 0.9999994039535522,
    #     "word": "住",
    #     "start": 8,
    #     "end": 9
    # },
    # {
    #     "entity_group": "P",
    #     "score": 0.9999999403953552,
    #     "word": "在",
    #     "start": 9,
    #     "end": 10
    # },
    # {
    #     "entity_group": "Nc",
    #     "score": 0.9999999403953552,
    #     "word": "柏",
    #     "start": 10,
    #     "end": 11
    # },
    # {
    #     "entity_group": "Nc",
    #     "score": 0.9891127943992615,
    #     "word": "林",
    #     "start": 11,
    #     "end": 12
    # },
    # {
    #     "entity_group": "PERIODCATEGORY",
    #     "score": 1,
    #     "word": "。",
    #     "start": 12,
    #     "end": 13
    # }
    # ]


def compare_ckiplab_ner(path="MODEL/ckiplab-bert-base-chinese-ner"):
    model = BertForTokenClassification.from_pretrained(path)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(path)
    text = "我叫克拉拉，我住在加州伯克利。"
    tokenized_text = tokenizer.tokenize(text)
    inputs = {
        k: paddle.to_tensor(
            v, dtype="int64").unsqueeze(0)
        for k, v in tokenizer(text).items()
    }
    with paddle.no_grad():
        score = paddle.nn.functional.softmax(model(**inputs), axis=-1)
    id2label = {
        "0": "O",
        "1": "B-CARDINAL",
        "2": "B-DATE",
        "3": "B-EVENT",
        "4": "B-FAC",
        "5": "B-GPE",
        "6": "B-LANGUAGE",
        "7": "B-LAW",
        "8": "B-LOC",
        "9": "B-MONEY",
        "10": "B-NORP",
        "11": "B-ORDINAL",
        "12": "B-ORG",
        "13": "B-PERCENT",
        "14": "B-PERSON",
        "15": "B-PRODUCT",
        "16": "B-QUANTITY",
        "17": "B-TIME",
        "18": "B-WORK_OF_ART",
        "19": "I-CARDINAL",
        "20": "I-DATE",
        "21": "I-EVENT",
        "22": "I-FAC",
        "23": "I-GPE",
        "24": "I-LANGUAGE",
        "25": "I-LAW",
        "26": "I-LOC",
        "27": "I-MONEY",
        "28": "I-NORP",
        "29": "I-ORDINAL",
        "30": "I-ORG",
        "31": "I-PERCENT",
        "32": "I-PERSON",
        "33": "I-PRODUCT",
        "34": "I-QUANTITY",
        "35": "I-TIME",
        "36": "I-WORK_OF_ART",
        "37": "E-CARDINAL",
        "38": "E-DATE",
        "39": "E-EVENT",
        "40": "E-FAC",
        "41": "E-GPE",
        "42": "E-LANGUAGE",
        "43": "E-LAW",
        "44": "E-LOC",
        "45": "E-MONEY",
        "46": "E-NORP",
        "47": "E-ORDINAL",
        "48": "E-ORG",
        "49": "E-PERCENT",
        "50": "E-PERSON",
        "51": "E-PRODUCT",
        "52": "E-QUANTITY",
        "53": "E-TIME",
        "54": "E-WORK_OF_ART",
        "55": "S-CARDINAL",
        "56": "S-DATE",
        "57": "S-EVENT",
        "58": "S-FAC",
        "59": "S-GPE",
        "60": "S-LANGUAGE",
        "61": "S-LAW",
        "62": "S-LOC",
        "63": "S-MONEY",
        "64": "S-NORP",
        "65": "S-ORDINAL",
        "66": "S-ORG",
        "67": "S-PERCENT",
        "68": "S-PERSON",
        "69": "S-PRODUCT",
        "70": "S-QUANTITY",
        "71": "S-TIME",
        "72": "S-WORK_OF_ART"
    }
    for t, s in zip(tokenized_text, score[0][1:-1]):
        index = paddle.argmax(s).item()
        label = id2label[str(index)]
        print(f"{label} {t} score {s[index].item()}")

    # O 我 score 0.9999998807907104
    # O 叫 score 1.0
    # B-PERSON 克 score 0.9999995231628418
    # I-PERSON 拉 score 0.9999992847442627
    # E-PERSON 拉 score 0.9999995231628418
    # O ， score 1.0
    # O 我 score 1.0
    # O 住 score 1.0
    # O 在 score 1.0
    # B-GPE 加 score 0.9999984502792358
    # I-GPE 州 score 0.9999964237213135
    # I-GPE 伯 score 0.9999923706054688
    # I-GPE 克 score 0.999998927116394
    # E-GPE 利 score 0.9999991655349731
    # O 。 score 0.9999994039535522

    # https://huggingface.co/ckiplab/bert-base-chinese-ner
    # [
    # {
    #     "entity_group": "PERSON",
    #     "score": 0.9999994039535522,
    #     "word": "克 拉",
    #     "start": 2,
    #     "end": 4
    # },
    # {
    #     "entity_group": "PERSON",
    #     "score": 0.9999994039535522,
    #     "word": "拉",
    #     "start": 4,
    #     "end": 5
    # },
    # {
    #     "entity_group": "GPE",
    #     "score": 0.9999964833259583,
    #     "word": "加 州 伯 克",
    #     "start": 9,
    #     "end": 13
    # },
    # {
    #     "entity_group": "GPE",
    #     "score": 0.9999991059303284,
    #     "word": "利",
    #     "start": 13,
    #     "end": 14
    # }
    # ]


if __name__ == "__main__":
    compare_ckiplab_ner()
    compare_ckiplab_pos()
    compare_ckiplab_ws()
    compare_math()
    compare_nlptown()
