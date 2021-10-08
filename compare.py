# coding:utf-8
import sys
from paddlenlp.transformers import RobertaForMaskedLM, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification
from paddlenlp.transformers import RobertaBPETokenizer, RobertaTokenizer
import paddle


def compare_math(path=""):
    path="E:\deep_learning\model_save\\roberta_base"

    model = RobertaForMaskedLM.from_pretrained(path)
    tokenizer = RobertaBPETokenizer.from_pretrained("roberta-base")
    text = ["The man worked as a","."]

    tokens_list=[]
    for i in range(2):
        tokens_list.append(tokenizer.tokenize(text[i]))

    tokens=['<s>']
    tokens.extend(tokens_list[0])
    tokens.extend(['<mask>'])
    tokens.extend(tokens_list[1])
    tokens.extend(['</s>'])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)

    model.eval()
    input_ids = paddle.to_tensor([token_ids])
    with paddle.no_grad():
        pd_outputs, sequence_output, hidden_state = model(input_ids)

    pd_outputs=pd_outputs[0]

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

    tokenizer = RobertaBPETokenizer.from_pretrained("E:\deep_learning\model_save\\roberta_large")
    questions=['Where do I live?']
    contexts = ['My name is Sarah and I live in London']

    token = tokenizer(questions, contexts, 
            stride=128, 
            max_seq_len=64,
            return_attention_mask=True,
            return_special_tokens_mask = True)
    print(token)
    special_tokens_mask = token[0]['special_tokens_mask']
    count = 3
    st_idx = 0
    for i in special_tokens_mask:
        st_idx+=1
        if i==1:
            count-=1
        if count==0:
           break
    
    input_ids=token[0]['input_ids']
    offset_mapping=token[0]['offset_mapping']

    input_ids = paddle.to_tensor(input_ids, dtype='int64').unsqueeze(0)

    model = RobertaForQuestionAnswering.from_pretrained('E:\deep_learning\model_save\deepset_roberta_base_squad2')
    model.eval()
    start, end = model(input_ids=input_ids)
    start_ = start[0][st_idx:-1].numpy()
    end_ = end[0][st_idx:-1].numpy()
    import numpy as np
    start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
    end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
    start_idx=offset_mapping[st_idx+start_.argmax()][0]
    end_idx=offset_mapping[st_idx+end_.argmax()][1]
    print(contexts[0][start_idx:end_idx])

def compare_uer_text_cls():
    tokenizer = RobertaTokenizer.from_pretrained("uer/roberta-base-finetuned-chinanews-chinese")
    token = tokenizer('北京上个月召开了两会')
    print(token)
    model = RobertaForSequenceClassification.from_pretrained('E:\deep_learning\model_save\\roberta-base-finetuned-chinanews-chinese')
    model.eval()
    input_ids = paddle.to_tensor(token['input_ids'], dtype='int64').unsqueeze(0)
    with paddle.no_grad():
        output = model(input_ids)
    import paddle.nn.functional as F
    output=F.softmax(output)
    id2label={
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
    tokenizer = RobertaTokenizer.from_pretrained("uer/roberta-base-finetuned-cluener2020-chinese")
    token = tokenizer(text)
    # print(token)
    model = RobertaForTokenClassification.from_pretrained('E:\deep_learning\model_save\\roberta-base-finetuned-cluener2020-chinese')
    model.eval()
    input_ids = paddle.to_tensor(token['input_ids'], dtype='int64').unsqueeze(0)
    with paddle.no_grad():
        output, hidden_state = model(input_ids)
    # paddle.save(output,'E:\deep_learning\代码\git_code\PaddleNLP\paddlenlp\\transformers\\roberta\output\\ner_output.pd')
    import paddle.nn.functional as F
    output=F.softmax(output)
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
    for t, s in zip(tokenized_text, output[0][1:-1]):
        index = paddle.argmax(s).item()
        label = id2label[str(index)]
        if index!=0:
            print(f"{label} {t} score {s[index].item()}")

def compare_uer_qa():
    tokenizer = RobertaTokenizer.from_pretrained("uer/roberta-base-chinese-extractive-qa")
    question = ['著名诗歌《假如生活欺骗了你》的作者是']
    context = ['普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。']
    token = tokenizer(question, context,
                    stride=256, 
                    max_seq_len=256)
    token_type_ids = token[0]["token_type_ids"]
    offset_mapping = token[0]["offset_mapping"]
    st_idx = len(token_type_ids)-sum(token_type_ids)
    

    model = RobertaForQuestionAnswering.from_pretrained('E:\deep_learning\model_save\\roberta-base-chinese-extractive-qa')
    model.eval()
    input_ids = paddle.to_tensor(token[0]['input_ids'], dtype='int64').unsqueeze(0)
    token_type_ids = paddle.to_tensor(token[0]['token_type_ids'], dtype='int64').unsqueeze(0)
    with paddle.no_grad():
        start, end = model(input_ids, token_type_ids)
    start_ = start[0][st_idx:-1].numpy()
    end_ = end[0][st_idx:-1].numpy()
    import numpy as np
    start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
    end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
    start_idx=offset_mapping[st_idx+start_.argmax()][0]
    end_idx=offset_mapping[st_idx+end_.argmax()][1]
    print(context[0][start_idx:end_idx])

def compare_distll_roberta_lm():
    path="E:\deep_learning\model_save\\tiny_disll_roberta"
    # path="E:\deep_learning\model_save\\roberta_base"

    model = RobertaForMaskedLM.from_pretrained(path)
    tokenizer = RobertaBPETokenizer.from_pretrained("roberta-base")
    text = ["The man worked as a","."]

    tokens_list=[]
    for i in range(2):
        tokens_list.append(tokenizer.tokenize(text[i]))

    tokens=['<s>']
    tokens.extend(tokens_list[0])
    tokens.extend(['<mask>'])
    tokens.extend(tokens_list[1])
    tokens.extend(['</s>'])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)

    model.eval()
    input_ids = paddle.to_tensor([token_ids])
    with paddle.no_grad():
        pd_outputs, sequence_output, hidden_state = model(input_ids)

    pd_outputs=pd_outputs[0]

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

if __name__ == "__main__":
    # compare_math()
    # compare_deepsort_qa()
    # compare_uer_clue_ner()
    # compare_uer_qa()
    # compare_distll_roberta_lm()
    compare_uer_text_cls()


   

    

    
