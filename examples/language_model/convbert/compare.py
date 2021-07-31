from paddlenlp.transformers import ConvBertTokenizer, ConvBertModel as PDConvBertModel
from transformers import ConvBertModel as PTConvBertModel
import paddle
import torch
paddle.set_device("cpu")

model_list = [
    ("YituTech/conv-bert-small", "convbert-small"),
    ("YituTech/conv-bert-medium-small", "convbert-medium-small"),
    ("YituTech/conv-bert-base", "convbert-base"),
]

for model in model_list:
    tokenizer = ConvBertTokenizer.from_pretrained(model[1])
    inputs = tokenizer("it is a nice day today!")["input_ids"]
    pt_inputs = torch.tensor([inputs])
    pd_inputs = paddle.to_tensor([inputs])

    pt_model = PTConvBertModel.from_pretrained(model[0])
    pt_model.eval()
    with torch.no_grad():
        pt_outputs = pt_model(pt_inputs)[0]

    pd_model = PDConvBertModel.from_pretrained(model[1])
    pd_model.eval()
    with paddle.no_grad():
        pd_outputs = torch.from_numpy(pd_model(pd_inputs).numpy())

    print(f"huggingface {model[0]} vs paddle {model[1]}")
    print("mean difference:", (pt_outputs - pd_outputs).abs().mean())
    print("max difference:", (pt_outputs - pd_outputs).abs().max())
