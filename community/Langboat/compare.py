import paddle
import paddlenlp.transformers as ppnlp
import torch
import transformers as hgnlp


def compare():
    bert = ppnlp.BertModel.from_pretrained('huhuiwen/mengzi-bert-base')
    tokenizer = ppnlp.BertTokenizer.from_pretrained('huhuiwen/mengzi-bert-base')
    inputs = tokenizer("Welcome to use PaddleNLP!")['input_ids']

    pt_inputs = torch.tensor([inputs])
    pd_inputs = paddle.to_tensor([inputs])

    pt_model = hgnlp.BertForMaskedLM.from_pretrained(
        'Langboat/mengzi-bert-base')
    pt_model.eval()
    with torch.no_grad():
        pt_outputs = pt_model(pt_inputs)[0]

    pd_model = ppnlp.BertForMaskedLM.from_pretrained(
        'huhuiwen/mengzi-bert-base')
    pd_model.eval()
    with paddle.no_grad():
        pd_outputs = torch.from_numpy(pd_model(pd_inputs).numpy())

    print(f"huggingface {'mengzi-bert-base'} vs paddle {'mengzi-bert-base'}")
    print("mean difference:", (pt_outputs - pd_outputs).abs().mean())
    print("max difference:", (pt_outputs - pd_outputs).abs().max())


def compare_fin():
    bert = ppnlp.BertModel.from_pretrained('huhuiwen/mengzi-bert-base-fin')
    tokenizer = ppnlp.BertTokenizer.from_pretrained(
        'huhuiwen/mengzi-bert-base-fin')
    inputs = tokenizer("Welcome to use PaddleNLP!")['input_ids']

    pt_inputs = torch.tensor([inputs])
    pd_inputs = paddle.to_tensor([inputs])

    pt_model = hgnlp.BertForMaskedLM.from_pretrained(
        'Langboat/mengzi-bert-base-fin')
    pt_model.eval()
    with torch.no_grad():
        pt_outputs = pt_model(pt_inputs)[0]

    pd_model = ppnlp.BertForMaskedLM.from_pretrained(
        'huhuiwen/mengzi-bert-base-fin')
    pd_model.eval()
    with paddle.no_grad():
        pd_outputs = torch.from_numpy(pd_model(pd_inputs).numpy())

    print(
        f"huggingface {'mengzi-bert-base-fin'} vs paddle {'mengzi-bert-base-fin'}"
    )
    print("mean difference:", (pt_outputs - pd_outputs).abs().mean())
    print("max difference:", (pt_outputs - pd_outputs).abs().max())


if __name__ == "__main__":

    compare()
    compare_fin()
