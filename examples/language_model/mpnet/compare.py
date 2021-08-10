from paddlenlp.transformers.mpnet.modeling import MPNetForMaskedLM as PDMPNetForMaskedLM
from transformers.models.mpnet.modeling_mpnet import (MPNetForMaskedLM as
                                                      PTMPNetForMaskedLM, )
import torch
import paddle

paddle.set_device("cpu")

pd_model = PDMPNetForMaskedLM.from_pretrained("mpnet-base")
pd_model.eval()
pt_model = PTMPNetForMaskedLM.from_pretrained("mpnet-base")
pt_model.eval()

with paddle.no_grad():
    pd_outputs = pd_model(
        paddle.to_tensor([[523, 123, 6123, 523, 5213, 632],
                          [5232, 1231, 6133, 5253, 5555, 6212]]))[0]

with torch.no_grad():
    pt_outputs = pt_model(
        torch.tensor([[523, 123, 6123, 523, 5213, 632],
                      [5232, 1231, 6133, 5253, 5555, 6212]]))[0]


def compare(a, b):
    a = torch.tensor(a.numpy()).float()
    b = torch.tensor(b.numpy()).float()
    meandif = (a - b).abs().mean()
    maxdif = (a - b).abs().max()
    print("mean difference:", meandif)
    print("max difference:", maxdif)


compare(pd_outputs, pt_outputs)
# meandif tensor(6.5154e-06)
# maxdif tensor(4.1485e-05)
