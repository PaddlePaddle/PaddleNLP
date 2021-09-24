from paddlenlp.transformers import GPTLMHeadModel as PDGPT2LMHeadModel, GPTTokenizer, BertTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as PTGPT2LMHeadModel
from paddlenlp.transformers import GPTForTokenClassification, GPTForSequenceClassification, GPTTokenizer
import paddle
import torch
import numpy as np

paddle.set_grad_enabled(False)
torch.set_grad_enabled(False)


def compare(a, b):
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    meandif = np.abs(a - b).mean()
    maxdif = np.abs(a - b).max()
    print("mean dif:", meandif)
    print("max dif:", maxdif)


def compare_lm(path="community/junnyu/microsoft-DialoGPT-small"):
    pdmodel = PDGPT2LMHeadModel.from_pretrained(path)
    ptmodel = PTGPT2LMHeadModel.from_pretrained(path).cuda()
    if "chinese" in path:
        text = "欢迎使用paddlenlp！"
        tokenizer = BertTokenizer.from_pretrained(path)
    else:
        text = "Welcome to paddlenlp!"
        tokenizer = GPTTokenizer.from_pretrained(path)
    pdmodel.eval()
    ptmodel.eval()
    pdinputs = {
        k: paddle.to_tensor(
            v, dtype="int64").unsqueeze(0)
        for k, v in tokenizer(
            text, return_token_type_ids=False).items()
    }
    ptinputs = {
        k: torch.tensor(
            v, dtype=torch.long).unsqueeze(0).cuda()
        for k, v in tokenizer(
            text, return_token_type_ids=False).items()
    }

    pd_logits = pdmodel(**pdinputs)

    pt_logits = ptmodel(**ptinputs).logits

    compare(pd_logits, pt_logits)


def test_GPTForTokenClassification():

    tokenizer = GPTTokenizer.from_pretrained("community/junnyu/distilgpt2")
    m = GPTForTokenClassification.from_pretrained("community/junnyu/distilgpt2")
    inputs = tokenizer(
        "Welcome to use PaddlePaddle and PaddleNLP!",
        return_token_type_ids=False)
    inputs = {
        k: paddle.to_tensor(
            [v], dtype="int64")
        for (k, v) in inputs.items()
    }
    logits = m(**inputs)
    print(logits.shape)


def test_GPTForSequenceClassification():
    paddle.set_grad_enabled(False)
    tokenizer = GPTTokenizer.from_pretrained("community/junnyu/distilgpt2")
    m = GPTForSequenceClassification.from_pretrained(
        "community/junnyu/distilgpt2")
    inputs = tokenizer(
        "Welcome to use PaddlePaddle and PaddleNLP!",
        return_token_type_ids=False)
    inputs = {
        k: paddle.to_tensor(
            [v], dtype="int64")
        for (k, v) in inputs.items()
    }
    logits = m(**inputs)
    print(logits.shape)


if __name__ == "__main__":
    # compare_lm(
    #     path="community/junnyu/microsoft-DialoGPT-small")
    # mean dif: 7.501994e-05
    # max dif: 0.00036621094
    # compare_lm(
    #     path="community/junnyu/distilgpt2")
    # mean dif: 7.249901e-06
    # max dif: 5.340576e-05
    # compare_lm(
    #     path="community/junnyu/uer-gpt2-chinese-poem")
    # mean dif: 1.0497178e-06
    # max dif: 1.335144e-05

    # test_GPTForTokenClassification()
    # [1, 13, 2]
    test_GPTForSequenceClassification()
    # [1, 2]
