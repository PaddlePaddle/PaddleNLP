from collections import OrderedDict
import numpy as np
import paddle
import torch
from paddlenlp.transformers import GPTJForCausalLM as PDGPTJModel
from transformers import GPTJForCausalLM as PTGPTJModel


def convert_pytorch_checkpoint_to_paddle(
        pytorch_checkpoint_path="./source/pytorch_model.bin",
        paddle_dump_path="model_state.pdparams",):

    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        print(f"Converting:{k}")
        paddle_state_dict[k] = v.data.numpy()
    paddle.save(paddle_state_dict, paddle_dump_path)


#
#
def compare(out_torch, out_paddle):
    out_torch = out_torch.detach().numpy()
    out_paddle = out_paddle.detach().numpy()
    print(f"out_torch.shape:{out_torch.shape}")
    print(f"out_paddle.shape:{out_paddle.shape}")
    assert out_torch.shape == out_paddle.shape
    abs_dif = np.abs(out_torch - out_paddle)
    mean_dif = np.mean(abs_dif)
    max_dif = np.max(abs_dif)
    min_dif = np.min(abs_dif)
    print("mean_dif:{}".format(mean_dif))
    print("max_dif:{}".format(max_dif))
    print("min_dif:{}".format(min_dif))


def test_forward():
    paddle.set_device("cpu")
    model_torch = PTGPTJModel.from_pretrained("./source/")
    model_paddle = PDGPTJModel.from_pretrained("./source/")
    model_torch.eval()
    model_paddle.eval()
    np.random.seed(42)
    x = np.random.randint(1, model_paddle.config["vocab_size"], size=(4, 64))

    input_torch = torch.tensor(x, dtype=torch.int64)
    out_torch = model_torch(input_torch)[0]
    # print(input_torch, out_torch)

    input_paddle = paddle.to_tensor(x, dtype=paddle.int64)
    out_paddle = model_paddle(input_paddle)[0]
    # print(input_paddle, out_paddle)

    print("torch result shape:{}".format(out_torch.shape))
    print("paddle result shape:{}".format(out_paddle.shape))
    compare(out_torch, out_paddle)
#
#
if __name__ == "__main__":
    convert_pytorch_checkpoint_to_paddle(
        "./source/pytorch_model.bin",
        "./source/model_state.pdparams")
    test_forward()
#     # torch result shape:torch.Size([4, 64, 30522])
#     # paddle result shape:[4, 64, 30522]
#     # mean_dif:1.666686512180604e-05
#     # max_dif:0.00015211105346679688
#     # min_dif:0.0
#

