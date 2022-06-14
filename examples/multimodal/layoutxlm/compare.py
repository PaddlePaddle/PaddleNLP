import sys
import paddle
import torch
import numpy as np

sys.path.insert(0, "../../../")


def get_input_demo(platform="paddle", device="cpu"):
    info = paddle.load("fake_input_paddle_xlm.data")
    # imgs = np.random.rand(info["input_ids"].shape[0], 3, 224, 224).astype(np.float32)
    # info["image"] = paddle.to_tensor(imgs)
    if platform == "torch":
        info = {key: torch.tensor(info[key].numpy()) for key in info}
        if device == "gpu":
            info = {key: info[key].cuda() for key in info}
    return info


def test_layoutlm_paddle():
    from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer

    tokenizer = LayoutXLMTokenizer.from_pretrained("layoutxlm-base-uncased")
    model = LayoutXLMModel.from_pretrained("layoutxlm-base-uncased")
    model.eval()

    paddle.save(model.state_dict(), "v2.pdparams")

    batch_input = get_input_demo(platform="paddle", device="gpu")
    with paddle.no_grad():
        outputs = model(
            input_ids=batch_input["input_ids"],
            bbox=batch_input["bbox"],
            image=batch_input["image"],
            attention_mask=batch_input["attention_mask"],
        )
    sequence_output = outputs[0]
    pooled_output = outputs[1]
    return sequence_output, pooled_output


def test_layoutlm_torch():
    # import pytorch models
    from layoutlmft.models.layoutxlm import LayoutXLMModel, LayoutXLMTokenizer
    model = LayoutXLMModel.from_pretrained("microsoft/layoutxlm-base")
    model.eval()
    model = model.cuda()

    batch_input = get_input_demo(platform="torch", device="gpu")

    outputs = model(
        input_ids=batch_input["input_ids"],
        bbox=batch_input["bbox"],
        image=batch_input["image"],
        attention_mask=batch_input["attention_mask"],
    )
    sequence_output = outputs[0]
    pooled_output = outputs[1]
    return sequence_output, pooled_output


def get_statistic_info(x, y):
    mean_abs_diff = np.mean(np.abs(x - y))
    max_abs_diff = np.max(np.abs(x - y))
    return mean_abs_diff, max_abs_diff


if __name__ == "__main__":

    print("\n====test_layoutxlm_torch=====")
    torch_hidden_out, torch_pool_out = test_layoutlm_torch()
    torch_hidden_out = torch_hidden_out.cpu().detach().numpy()
    torch_pool_out = torch_pool_out.cpu().detach().numpy()
    print(torch_hidden_out.shape, torch_pool_out.shape)

    print("\n====test_layoutxlm_paddle=====")
    paddle_hidden_out, paddle_pool_out = test_layoutlm_paddle()
    paddle_hidden_out = paddle_hidden_out.numpy()
    paddle_pool_out = paddle_pool_out.numpy()
    print(paddle_hidden_out.shape, paddle_pool_out.shape)

    mean_abs_diff, max_abs_diff = get_statistic_info(torch_hidden_out,
                                                     paddle_hidden_out)
    print("======hidden_out diff  info====")
    print("\t mean_abs_diff: {}".format(mean_abs_diff))
    print("\t max_abs_diff: {}".format(max_abs_diff))

    mean_abs_diff, max_abs_diff = get_statistic_info(torch_pool_out,
                                                     paddle_pool_out)
    print("======pool_out diff  info====")
    print("\t mean_abs_diff: {}".format(mean_abs_diff))
    print("\t max_abs_diff: {}".format(max_abs_diff))
